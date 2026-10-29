class StateMachine():
    '''a class containing state machine functionality'''
    def __init__(self):
        pass

    def change_state(self, state):
        '''changes the state and puts the new state into its run loop'''
        self.state.exit()
        self.state = state
        self.state.enter()

    def runtime(self):
        ''''''
        self.state.tick()