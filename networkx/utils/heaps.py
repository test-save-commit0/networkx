"""
Min-heaps.
"""
from heapq import heappop, heappush
from itertools import count
import networkx as nx
__all__ = ['MinHeap', 'PairingHeap', 'BinaryHeap']


class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """


    class _Item:
        """Used by subclassess to represent a key-value pair."""
        __slots__ = 'key', 'value'

        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __repr__(self):
            return repr((self.key, self.value))

    def __init__(self):
        """Initialize a new min-heap."""
        self._dict = {}

    def min(self):
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        if not self._dict:
            raise nx.NetworkXError("The heap is empty.")
        min_item = min(self._dict.values(), key=lambda x: x.value)
        return min_item.key, min_item.value

    def pop(self):
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        if not self._dict:
            raise nx.NetworkXError("The heap is empty.")
        min_item = min(self._dict.values(), key=lambda x: x.value)
        del self._dict[min_item.key]
        return min_item.key, min_item.value

    def get(self, key, default=None):
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
        item = self._dict.get(key)
        return item.value if item else default

    def insert(self, key, value, allow_increase=False):
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
        if key in self._dict:
            if allow_increase or value < self._dict[key].value:
                self._dict[key].value = value
                return True
            return False
        else:
            self._dict[key] = self._Item(key, value)
            return True

    def __nonzero__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __bool__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __len__(self):
        """Returns the number of key-value pairs in the heap."""
        return len(self._dict)

    def __contains__(self, key):
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """
        return key in self._dict


class PairingHeap(MinHeap):
    """A pairing heap."""


    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """
        __slots__ = 'left', 'next', 'prev', 'parent'

        def __init__(self, key, value):
            super().__init__(key, value)
            self.left = None
            self.next = None
            self.prev = None
            self.parent = None

    def __init__(self):
        """Initialize a pairing heap."""
        super().__init__()
        self._root = None

    def decrease_key(self, key, new_value):
        """Decrease the value associated with a key."""
        if key not in self._dict:
            raise KeyError(f"Key {key} not found in the heap")
        node = self._dict[key]
        if new_value >= node.value:
            return False
        self._cut(node)
        node.value = new_value
        if self._root:
            self._root = self._link(self._root, node)
        else:
            self._root = node
        return True

    def delete(self, key):
        """Delete a key-value pair from the heap."""
        if key not in self._dict:
            raise KeyError(f"Key {key} not found in the heap")
        node = self._dict[key]
        self._cut(node)
        new_tree = self._merge_children(node)
        if self._root == node:
            self._root = new_tree
        else:
            if new_tree:
                self._root = self._link(self._root, new_tree)
        del self._dict[key]

    def merge(self, other):
        """Merge another PairingHeap into this one."""
        if not isinstance(other, PairingHeap):
            raise TypeError("Can only merge with another PairingHeap")
        if other._root:
            if self._root:
                self._root = self._link(self._root, other._root)
            else:
                self._root = other._root
            self._dict.update(other._dict)
        other._root = None
        other._dict.clear()

    def min(self):
        if not self._root:
            raise nx.NetworkXError("The heap is empty.")
        return self._root.key, self._root.value

    def pop(self):
        if not self._root:
            raise nx.NetworkXError("The heap is empty.")
        min_node = self._root
        self._root = self._merge_children(self._root)
        del self._dict[min_node.key]
        return min_node.key, min_node.value

    def insert(self, key, value, allow_increase=False):
        if key in self._dict:
            node = self._dict[key]
            if allow_increase or value < node.value:
                self._cut(node)
                node.value = value
                self._root = self._link(self._root, node)
                return True
            return False
        else:
            new_node = self._Node(key, value)
            self._dict[key] = new_node
            if self._root:
                self._root = self._link(self._root, new_node)
            else:
                self._root = new_node
            return True

    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
        if root.value <= other.value:
            other.parent = root
            other.prev = root.left
            if root.left:
                root.left.next = other
            other.next = None
            root.left = other
            return root
        else:
            root.parent = other
            root.prev = other.left
            if other.left:
                other.left.next = root
            root.next = None
            other.left = root
            return other

    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
        if not root.left:
            return None
        
        # First pass: link siblings in pairs
        current = root.left
        next_node = None
        first_pass = []
        while current:
            next_node = current.next
            current.next = current.prev = current.parent = None
            if next_node:
                next_node.next = next_node.prev = next_node.parent = None
                first_pass.append(self._link(current, next_node))
                current = next_node.next
            else:
                first_pass.append(current)
                break
        
        # Second pass: link the results of the first pass
        while len(first_pass) > 1:
            first_pass.append(self._link(first_pass.pop(0), first_pass.pop(0)))
        
        return first_pass[0] if first_pass else None

    def _cut(self, node):
        """Cut a node from its parent."""
        if node.parent:
            if node.parent.left == node:
                node.parent.left = node.next
            if node.prev:
                node.prev.next = node.next
            if node.next:
                node.next.prev = node.prev
            node.next = node.prev = node.parent = None


class BinaryHeap(MinHeap):
    """A binary heap."""

    def __init__(self):
        """Initialize a binary heap."""
        super().__init__()
        self._heap = []
        self._count = count()
