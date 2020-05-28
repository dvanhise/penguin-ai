# Mock agent that requires human input for moves
class HumanAgent:
    def step(self, state, player, training=True):
        target = destination = None
        if state['phase'] == 0:
            while target not in state['placements']:
                print(state['placements'])
                args = input('Place a penguin: ')
                try:
                    target = tuple([int(arg) for arg in args.split(' ')])
                    print(target)
                except:
                    print('Invalid')
                    target = None
        elif state['phase'] == 1:
            while target not in state['moves']:
                print(state['moves'].keys())
                args = input('Choose a penguin: ')
                try:
                    target = tuple([int(arg) for arg in args.split(' ')])
                except:
                    print('Invalid')
                    target = None

            while destination not in state['moves'][target]:
                print(state['moves'][target])
                args = input('Choose destination: ')
                try:
                    destination = tuple([int(arg) for arg in args.split(' ')])
                except:
                    print('Invalid')
                    destination = None

        return target, destination

    def step_end(self, *args, **kwargs):
        pass
