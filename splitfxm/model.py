class Model:
    def __init__(self):
        """
        Initialize a Model object.
        """
        self._equation = None

    def equation(self):
        """
        Get the equation associated with this model

        Returns
        -------
        list
            The list of equations
        """
        return self._equation
