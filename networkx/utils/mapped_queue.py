"""Priority queue class with updatable priorities.
"""
import heapq
__all__ = ['MappedQueue']


class _HeapElement:
    """This proxy class separates the heap element from its priority.

    The idea is that using a 2-tuple (priority, element) works
    for sorting, but not for dict lookup because priorities are
    often floating point values so round-off can mess up equality.

    So, we need inequalities to look at the priority (for sorting)
    and equality (and hash) to look at the element to enable
    updates to the priority.

    Unfortunately, this class can be tricky to work with if you forget that
    `__lt__` compares the priority while `__eq__` compares the element.
    In `greedy_modularity_communities()` the following code is
    used to check that two _HeapElements differ in either element or priority:

        if d_oldmax != row_max or d_oldmax.priority != row_max.priority:

    If the priorities are the same, this implementation uses the element
    as a tiebreaker. This provides compatibility with older systems that
    use tuples to combine priority and elements.
    """
    __slots__ = ['priority', 'element', '_hash']

    def __init__(self, priority, element):
        self.priority = priority
        self.element = element
        self._hash = hash(element)

    def __lt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority < other
        if self.priority == other_priority:
            try:
                return self.element < other.element
            except TypeError as err:
                raise TypeError(
                    'Consider using a tuple, with a priority value that can be compared.'
                    )
        return self.priority < other_priority

    def __gt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority > other
        if self.priority == other_priority:
            try:
                return self.element > other.element
            except TypeError as err:
                raise TypeError(
                    'Consider using a tuple, with a priority value that can be compared.'
                    )
        return self.priority > other_priority

    def __eq__(self, other):
        try:
            return self.element == other.element
        except AttributeError:
            return self.element == other

    def __hash__(self):
        return self._hash

    def __getitem__(self, indx):
        return self.priority if indx == 0 else self.element[indx - 1]

    def __iter__(self):
        yield self.priority
        try:
            yield from self.element
        except TypeError:
            yield self.element

    def __repr__(self):
        return f'_HeapElement({self.priority}, {self.element})'


class MappedQueue:
    """The MappedQueue class implements a min-heap with removal and update-priority.

    The min heap uses heapq as well as custom written _siftup and _siftdown
    methods to allow the heap positions to be tracked by an additional dict
    keyed by element to position. The smallest element can be popped in O(1) time,
    new elements can be pushed in O(log n) time, and any element can be removed
    or updated in O(log n) time. The queue cannot contain duplicate elements
    and an attempt to push an element already in the queue will have no effect.

    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it adds element removal, lookup, and priority update.

    Parameters
    ----------
    data : dict or iterable

    Examples
    --------

    A `MappedQueue` can be created empty, or optionally, given a dictionary
    of initial elements and priorities.  The methods `push`, `pop`,
    `remove`, and `update` operate on the queue.

    >>> colors_nm = {"red": 665, "blue": 470, "green": 550}
    >>> q = MappedQueue(colors_nm)
    >>> q.remove("red")
    >>> q.update("green", "violet", 400)
    >>> q.push("indigo", 425)
    True
    >>> [q.pop().element for i in range(len(q.heap))]
    ['violet', 'indigo', 'blue']

    A `MappedQueue` can also be initialized with a list or other iterable. The priority is assumed
    to be the sort order of the items in the list.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> [q.pop() for i in range(len(q.heap))]
    [50, 916, 1117, 4609]

    An exception is raised if the elements are not comparable.

    >>> q = MappedQueue([100, "a"])
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'int' and 'str'

    To avoid the exception, use a dictionary to assign priorities to the elements.

    >>> q = MappedQueue({100: 0, "a": 1})

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=None):
        """Priority queue class with updatable priorities."""
        if data is None:
            self.heap = []
        elif isinstance(data, dict):
            self.heap = [_HeapElement(v, k) for k, v in data.items()]
        else:
            self.heap = list(data)
        self.position = {}
        self._heapify()

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        pass

    def __len__(self):
        return len(self.heap)

    def push(self, elt, priority=None):
        """Add an element to the queue."""
        pass

    def pop(self):
        """Remove and return the smallest element in the queue."""
        pass

    def update(self, elt, new, priority=None):
        """Replace an element in the queue with a new one."""
        pass

    def remove(self, elt):
        """Remove an element from the queue."""
        pass

    def _siftup(self, pos):
        """Move smaller child up until hitting a leaf.

        Built to mimic code for heapq._siftup
        only updating position dict too.
        """
        pass

    def _siftdown(self, start_pos, pos):
        """Restore invariant. keep swapping with parent until smaller.

        Built to mimic code for heapq._siftdown
        only updating position dict too.
        """
        pass
