from numpy import (
    ndarray,
    uint8,
    float32,
    float64,
    nonzero,
    median,
    column_stack,
    ones_like,
    mgrid,
    where,
    abs,
    array,
    partition,
    unique,
    zeros,
    empty,
    arange,
)
from numpy.linalg import lstsq
import cv2
from logging import info
from dependencies.row_blocks import run_in_row_blocks

class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''
    def __init__(self, pixel_pitch_mm: list, plate_tolerance_mm: float = 3.0, min_subject_height_mm: float = 5.0, gap_close_px: int = 31, depth_support_px: int = 50, max_glare_fraction: float = 0.2):
        '''
        Args:
            pixel_pitch_mm: [x, y] size of one image pixel in mm
            plate_tolerance_mm: how far from the fitted plate a pixel can be and still count as plate
            min_subject_height_mm: how far above the plate a pixel must be to count as the subject
            gap_close_px: dropout gaps in the subject smaller than this are closed before outlining it
            depth_support_px: depth is the height reached by at least this many pixels, so a few
                stray pixels of glare cannot set it
            max_glare_fraction: frames where more than this share of the subject's readings were
                glare are rejected
        '''
        if pixel_pitch_mm is None or len(pixel_pitch_mm) != 2 or min(pixel_pitch_mm) <= 0:
            raise ValueError(f"Error : pixel pitch of {pixel_pitch_mm} is not valid, must be [x, y] in mm and above 0")
        self.pixel_pitch_mm = array(pixel_pitch_mm, dtype=float32)
        self.plate_tolerance_mm = plate_tolerance_mm
        self.min_subject_height_mm = min_subject_height_mm
        self.gap_close_px = gap_close_px
        self.depth_support_px = depth_support_px
        self.max_glare_fraction = max_glare_fraction
        self.height_image = None
        self.binary_image = None

    def _get_plate_heights(self, image: ndarray, valid: ndarray, step: int = 8) -> ndarray:
        '''fits a plane to the plate the subject sits on and returns each pixel's height above it
        Args:
            image: an ndarray representing the depth image in mm
            valid: an ndarray of bools marking the pixels that hold a measurement
            step: only every step-th pixel in x and y is used for the fit
        Returns:
            heights: an ndarray of heights above the plate in mm, 0 where invalid
        '''
        ys, xs = nonzero(valid[::step, ::step])
        ys, xs = ys * step, xs * step
        z = image[ys, xs].astype(float64)
        points = column_stack([xs, ys, ones_like(xs)]).astype(float64)
        # the plate is the lowest large surface, so start from the lower half and refine
        plate = z <= median(z)
        for _ in range(6):
            coefficients = lstsq(points[plate], z[plate], rcond=None)[0]
            plate = abs(z - points @ coefficients) < self.plate_tolerance_mm
        # the plane a*x + b*y + c, evaluated by broadcasting a row of x against a column of y
        # (the same arithmetic as full coordinate grids, without building them), in row blocks
        heights = empty(image.shape, float32)
        x = arange(image.shape[1])[None, :]

        def block(top, bottom):
            y = arange(top, bottom)[:, None]
            plane = coefficients[0] * x + coefficients[1] * y + coefficients[2]
            heights[top:bottom] = where(valid[top:bottom], image[top:bottom] - plane, 0)

        run_in_row_blocks(image.shape[0], block)
        return heights

    @staticmethod
    def _enclosed_dropouts(valid: ndarray, raised: ndarray) -> ndarray:
        '''the pixels with no reading that sit inside the floor and against the subject
        A dark or glossy plate returns almost no laser light, so most of it has no reading at all;
        the floor around it still does, so the plate shows as a hole in the floor. Holes that touch
        the edge of the image are scan margins, and holes nowhere near anything raised (a dark mark
        on an empty floor) are not a plate, so both are left out.
        Args:
            valid: an ndarray of bools marking the pixels that hold a measurement
            raised: an ndarray of bools marking the pixels above the subject height
        Returns:
            an ndarray of bools marking the dropouts that belong to the subject
        '''
        count, labels, stats, _ = cv2.connectedComponentsWithStats((~valid).astype(uint8), connectivity=8)
        h, w = valid.shape
        x, y, bw, bh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
        on_border = (x == 0) | (y == 0) | (x + bw >= w) | (y + bh >= h)
        near_raised = cv2.dilate(raised.astype(uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        touching = zeros(count, bool)
        touching[labels[near_raised & ~valid]] = True     # marks each label present, no sort needed
        keep = touching & ~on_border
        keep[0] = False  # label 0 is the pixels that do have a reading
        return keep[labels]

    def _get_largest_subject(self, valid: ndarray = None):
        '''finds the largest region standing above the plate and returns its outline
        Args:
            valid: the pixels that hold a measurement; when given, dropouts enclosed by the floor
                and against the subject count as subject, so a plate that returns little laser
                light is outlined by its footprint rather than by the few readings it gave
        Returns:
            subject_hull: the convex hull of the largest region in pixels, or None if there is none
        '''
        raised = self.height_image > self.min_subject_height_mm
        if not raised.any():
            info("INFO : No subject found above the plate")
            return None
        if valid is not None:
            raised = raised | self._enclosed_dropouts(valid, raised)
        subject = raised.astype(uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.gap_close_px, self.gap_close_px))
        # Close only around the subject: with a margin of twice the kernel size every pixel the
        # closing can change is inside the window, and the window edge sees only zeros, so the
        # result is the same as closing the whole image.
        rows, cols = nonzero(subject.any(axis=1))[0], nonzero(subject.any(axis=0))[0]
        margin = 2 * self.gap_close_px
        top, bottom = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, subject.shape[0])
        left, right = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, subject.shape[1])
        self.binary_image = zeros(subject.shape, uint8)
        self.binary_image[top:bottom, left:right] = cv2.morphologyEx(
            subject[top:bottom, left:right], cv2.MORPH_CLOSE, kernel
        ) * 255
        contours, hierachy = cv2.findContours(
            self.binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            info("INFO : No subject found above the plate")
            return None
        # the hull ignores bites taken out of the outline by dropouts along the rim
        return cv2.convexHull(max(contours, key=cv2.contourArea))

    def get_subject_details(self, image: ndarray, glare_mask: ndarray = None) -> dict:
        '''Extracts the depth, perimeter and radius of the subject in mm
        Args:
            image: a np.ndarray representing the depth image in mm, 0 marks invalid pixels
            glare_mask: the pixels already removed from the image as glare, used to reject the
                frame when too much of the subject was glare
        Returns:
            details: a dictionary containing details of the subject, or False if there is no
                subject or the frame is rejected for glare'''
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        valid = single_channel > 0
        if not valid.any():
            info("INFO : No valid pixels in image")
            return False
        self.height_image = self._get_plate_heights(single_channel, valid)

        subject_hull = self._get_largest_subject(valid)
        if subject_hull is None:
            return False
        # everything below only concerns pixels inside the hull, so work within its bounding box
        left, top, box_w, box_h = cv2.boundingRect(subject_hull)
        box = (slice(top, top + box_h), slice(left, left + box_w))
        inside = zeros((box_h, box_w), uint8)
        inside = cv2.drawContours(inside, [subject_hull - array([left, top], dtype=subject_hull.dtype)], -1, 1, -1) > 0
        valid_box = valid[box]
        if glare_mask is not None:
            # as a share of the pixels that had a reading, dropouts would dilute it otherwise
            glare_box = glare_mask[box]
            glare_fraction = (glare_box & inside).sum() / (inside & (valid_box | glare_box)).sum()
            if glare_fraction > self.max_glare_fraction:
                info(f"INFO : frame rejected, {glare_fraction:.0%} of the subject was glare")
                return False
        hull_mm = (subject_hull.reshape(-1, 2) * self.pixel_pitch_mm).astype(float32)
        _, radius = cv2.minEnclosingCircle(hull_mm)
        subject_heights = self.height_image[box][inside & valid_box]
        support = min(self.depth_support_px, subject_heights.size)

        details = dict()
        details["depth"] = float(partition(subject_heights, -support)[-support])
        details["perimeter"] = float(cv2.arcLength(hull_mm, True))
        details["radius"] = float(radius)
        return details
