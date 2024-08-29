class Model:
    def __init__(self):
        """
        Initialize a Model object.
        """
        self._equations = []

    def equations(self):
        """
        Get the list of equations associated with this Model.

        Returns
        -------
        list
            The list of equations
        """
        return self._equations
