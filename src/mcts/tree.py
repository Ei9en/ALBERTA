import math


class MCTSTree:

    def __init__(
        self,
        root,
        c_puct=1.5,
    ):
        self.root = root

        self.c_puct = c_puct


    def select_child(
        self,
        node,
    ):

        best_child = None

        best_score = float("-inf")


        sqrt_visits = math.sqrt(
            max(
                node.visit_count,
                1,
            )
        )


        for child in node.children.values():

            q = child.value

            u = (
                self.c_puct
                * child.prior
                * sqrt_visits
                / (
                    1
                    + child.visit_count
                )
            )


            score = q + u


            if score > best_score:

                best_score = score

                best_child = child


        return best_child


    def backup(
        self,
        node,
        value,
    ):

        current = node

        current_value = value


        while current is not None:

            current.visit_count += 1

            current.value_sum += (
                current_value
            )


            current_value = (
                -current_value
            )


            current = (
                current.parent
            )