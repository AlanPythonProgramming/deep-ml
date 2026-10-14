from __future__ import annotations   # at the very top of the file
from dataclasses import dataclass
from typing import Optional

@dataclass
class Node:
    errors: float
    leaf_errors: float
    num_leaves: int
    left: Optional[Node] = None
    right: Optional[Node] = None
    
def stat_calcs(tree: dict) -> Node:
    if tree['left'] is None and tree['right'] is None:
        return Node(tree['errors'], tree['errors'], 1)
    leaf_errors = 0
    num_leaves = 0
    left, right = None, None
    if tree['left'] is not None:
        left = stat_calcs(tree['left'])
        leaf_errors += left.leaf_errors
        num_leaves += left.num_leaves
    if tree['right'] is not None:
        right = stat_calcs(tree['right'])
        leaf_errors += right.leaf_errors
        num_leaves += right.num_leaves
    return Node(tree['errors'], leaf_errors, num_leaves, left, right)

def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    root = stat_calcs(tree)
    alphas = []
    nodes_stack = [root]
    while len(nodes_stack):
        node = nodes_stack.pop()
        if node.left is None and node.right is None:
            continue
        alpha = (node.errors - node.leaf_errors) / (node.num_leaves-1)
        alphas.append(alpha)
        if node.left is not None:
            nodes_stack.append(node.left)
        if node.right is not None:
            nodes_stack.append(node.right)

    return sorted(alphas)    