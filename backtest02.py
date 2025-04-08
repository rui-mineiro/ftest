
from abc import ABC, abstractmethod

class Queue(ABC):
  """ API for type Queue """

  @abstractmethod
  def isEmpty(self):
    """returns true iff queue is empty"""

  @abstractmethod
  def enqueue(self, item):
    """inserts item at queue's end"""

  @abstractmethod
  def front(self):
    """requires: not isEmpty()
       returns the item at queue's beginning"""

  @abstractmethod
  def dequeue(self):
    """requires: not isEmpty()
       removes the item at queue's beginning"""



class CircularArrayQueue(Queue):
  """ representing a queue with a circular array """

  def __init__(self, capacity=6):
    self._queue = [None]*capacity
    self._begin = 0
    self._end = 0
    self._size = 0

  def isEmpty(self):
    return self._size == 0

  def enqueue(self, item):
    if self._size == len(self._queue): # queue is full, double array size
      self._reallocate()
    self._queue[self._end] = item
    self._end = self._inc(self._end)
    self._size += 1

  def front(self):
    return self._queue[self._begin]

  def dequeue(self):
    self._queue[self._begin] = None
    self._begin = self._inc(self._begin)
    self._size -= 1

  def _inc(self, n):
    """ increment by 1, using modular arithmetic """
    return (n+1)%len(self._queue)

  def _reallocate(self):
    """ auxiliary method called when queue is full """
    newQueue = [None] * (2*self._size)
    j = self._begin
    for i in range(self._size):
      newQueue[i] = self._queue[j]
      j = self._inc(j)
    self._begin = 0
    self._end   = self._size
    self._queue = newQueue

  def __len__(self):
    return self._size

  def __repr__(self):
    result = []
    j = self._begin
    for i in range(self._size):
      result.append(self._queue[j])
      j = self._inc(j)
    return '<'+str(result)[1:-1]+'<'

def bincount():
  q = CircularArrayQueue()
  q.enqueue('1')
  while True:
    current = q.front()
    q.dequeue()
    yield current
    q.enqueue(current+'0')
    q.enqueue(current+'1')
    a=1

gen = bincount()
print([next(gen) for _ in range(8)])    

