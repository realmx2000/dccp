__author__ = 'Xinyue'
import numpy as np
import cvxpy as cvx

def linearize_para(expr):
    '''
    input:
        expr: an expression
    return:
        linear_expr: linearized expression
        zero_order: zero order parameter
        linear_dictionary: {variable: [value parameter, [gradient parameter]]}
        dom: domain
    '''
    zero_order = cvx.Parameter(expr.shape[0],expr.shape[1]) # zero order
    linear_expr = zero_order
    linear_dictionary = {}
    for var in expr.variables():
        value_para = cvx.Parameter(var.shape[0],var.shape[1])
        if var.ndim > 1: # matrix to vector
            gr = []
            for d in range(var.shape[1]):
                g = cvx.Parameter(var.shape[0],expr.shape[0])
                # g = g.T
                linear_expr += g.T * (var[:,d] - value_para[:,d]) # first order
                gr.append(g)
            linear_dictionary[var] = [value_para, gr]
        else: # vector to vector
            g = cvx.Parameter(var.shape[0],expr.shape[0])
            linear_expr += g.T * (var[:,d] - value_para[:,d]) # first order
            gr.append(g)
        linear_dictionary[var] = [value_para, gr]
    dom = expr.domain
    return linear_expr, zero_order, linear_dictionary, dom

def linearize(expr, vars=None, grads=None):
    """Returns the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

    Args:
        expr: An expression.

    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    else:
        if expr.value is None:
            raise ValueError(
        "Cannot linearize non-affine expression with missing variable values."
            )
        if grads is None:
            grads = {}

        base_key = str(expr) + 'value'
        if base_key in grads:
            grads[base_key].value = expr.value  # Not a grad, but stored here for convenience
        else:
            grads[base_key] = cvx.Parameter(expr.value.shape, value=expr.value)
        tangent = grads[base_key]

        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            if vars is not None:
                vars[var.name()].value = var.value
            key = str(expr) + var.name()
            if key in grads:
                grads[key].value = grad_map[var]
            else:
                grads[key] = cvx.Parameter(grad_map[var].shape, value=grad_map[var])
                value = var.value if vars is None else vars[var.name()]

                if var.ndim > 1:
                    temp = cvx.reshape(cvx.vec(var - value), (var.shape[0] * var.shape[1], 1))
                    flattened = grads[key].T @ temp
                    tangent = tangent + cvx.reshape(flattened, expr.shape)
                    tangent = tangent + cvx.reshape(flattened, expr.shape)
                else:
                    tangent = tangent + grads[key].T @ (var - value)
        return tangent