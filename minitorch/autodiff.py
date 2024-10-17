from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    shifted_up, shifted_down = list(vals), list(vals)
    shifted_up[arg] += epsilon
    shifted_down[arg] -= epsilon
    delta = f(*shifted_up) - f(*shifted_down)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result: List[Variable] = []

    def visit(var: Variable) -> None:
        if var.is_constant():  # pass constant
            return
        if var.unique_id in visited:  # has been visited
            return
        if not var.is_leaf():
            for input in var.parents:
                visit(input)
        visited.add(var.unique_id)
        result.insert(0, var)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = topological_sort(variable)
    var2deriv = {}
    var2deriv[variable.unique_id] = deriv
    for var in queue:
        if var.is_leaf():
            continue
        if var.unique_id in var2deriv.keys():
            deriv = var2deriv[var.unique_id]
        deriv_tmp = var.chain_rule(deriv)
        for input, input_deriv in deriv_tmp:
            if input.is_leaf():
                input.accumulate_derivative(input_deriv)
                continue
            if input.unique_id in var2deriv.keys():
                var2deriv[input.unique_id] += input_deriv
            else:
                var2deriv[input.unique_id] = input_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
