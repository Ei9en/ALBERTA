class MCTSNode:

    def __init__(
        self,
        board,
        parent=None,
        action=None,
        prior=0.0,
    ):
        self.board = board

        self.parent = parent

        self.action = action

        self.prior = float(prior)

        self.visit_count = 0

        self.value_sum = 0.0

        self.children = {}

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum
            / self.visit_count
        )

    @property
    def is_expanded(self):

        return len(self.children) > 0

    @property
    def is_terminal(self):

        return self.board.is_game_over()