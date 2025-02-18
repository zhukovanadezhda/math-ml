import math

class Vector:
    def __init__(self, data):
        """
        Initialize a Vector with a list of numerical data.
        :param data: List of numbers representing the vector.
        """
        if not isinstance(data, list):
            raise TypeError("Data should be a list of numbers.")
        if not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("All elements in the data should be numbers.")
        self.data = data

    def __len__(self):
        """Return the dimension (length) of the vector."""
        return len(self.data)
    
    def __neg__(self):
        """Return the negation of the vector."""
        return Vector([-x for x in self.data])

    def __add__(self, other):
        """
        Element-wise addition of two vectors.
        :param other: Another Vector of the same dimension.
        :return: A new Vector representing the sum.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for addition.")
        return Vector([a + b for a, b in zip(self.data, other.data)])
    
    def __sub__(self, other):
        """
        Element-wise subtraction of two vectors.
        :param other: Another Vector of the same dimension.
        :return: A new Vector representing the difference.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for subtraction.")
        return Vector([a - b for a, b in zip(self.data, other.data)])
    
    def __mul__(self, other):
        """
        Scalar multiplication of a vector.
        :param other: A scalar (int or float).
        :return: A new Vector representing the product.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Multiplication is only supported with a scalar.")
        return Vector([a * other for a in self.data])

    def __rmul__(self, other):
        """Allow scalar multiplication from the left."""
        return self.__mul__(other)

    def __eq__(self, other):
        """
        Check if two vectors are equal.
        :param other: Another Vector.
        :return: True if they have the same elements, False otherwise.
        """
        if not isinstance(other, Vector) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self.data, other.data))

    def __iter__(self):
        """Allow iteration over the vector's elements."""
        return iter(self.data)

    def __getitem__(self, index):
        """Allow indexing to access vector elements."""
        return self.data[index]

    def __str__(self):
        """Return a user-friendly string representation of the vector."""
        return str(self.data)
    
    def dot(self, other):
        """
        Compute the dot product of two vectors.
        :param other: Another Vector of the same dimension.
        :return: The dot product (a number).
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for dot product.")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self):
        """
        Compute the L2 norm (Euclidean norm) of the vector.
        :return: The Euclidean norm.
        """
        return math.sqrt(sum(x**2 for x in self.data))

    def normalize(self):
        """
        Return a normalized (unit length) vector.
        :return: A new Vector that is the normalized version of this vector.
        """
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return Vector([x / n for x in self.data])
