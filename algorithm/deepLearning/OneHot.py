import numpy

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes

  labels_one_hot = numpy.zeros((num_labels, num_classes))

  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  print(labels_one_hot)
  return labels_one_hot


test_matrix = numpy.array([3,4,8])
dense_to_one_hot(test_matrix)
