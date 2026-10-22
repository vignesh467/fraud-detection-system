"""The Transaction class: one card payment. Shows ENCAPSULATION."""


class Transaction:
    """Represents a single card transaction.

    The attributes start with two underscores (private), so other code
    cannot change them directly. It must go through the properties below,
    which can check that the values are valid.
    """

    def __init__(self, transaction_id, amount, time, features):
        self.__transaction_id = transaction_id
        self.__time = float(time)
        self.__features = dict(features)   # V1 ... V28
        self.amount = amount                # goes through the setter (validation)

    # ----- Getters (read access) -----
    @property
    def transaction_id(self):
        return self.__transaction_id

    @property
    def time(self):
        return self.__time

    @property
    def amount(self):
        return self.__amount

    # ----- Setter (controlled write access) -----
    @amount.setter
    def amount(self, value):
        if value < 0:
            raise ValueError("Amount cannot be negative")
        self.__amount = float(value)

    def to_dict(self):
        """Return the values in the same column order as the dataset."""
        return {"Time": self.__time, **self.__features, "Amount": self.__amount}

    @classmethod
    def from_row(cls, transaction_id, row):
        """Build a Transaction from one row of the pandas DataFrame."""
        features = {col: row[col] for col in row.index if col.startswith("V")}
        return cls(transaction_id, row["Amount"], row["Time"], features)

    def __repr__(self):
        return f"Transaction(id={self.__transaction_id}, amount={self.__amount:.2f})"

if __name__ == "__main__":
    # 1. Create a valid transaction (features is an empty dict for now)
        t = Transaction(1, 120.5, 0, {})

        # 2. Print it: this calls your __repr__ method
        print(t)

        # 3. Read values through the properties
        print("ID:", t.transaction_id)
        print("Amount:", t.amount)

        # 4. Change the amount through the setter (allowed, because it is valid)
        t.amount = 200
        print("New amount:", t.amount)

        # 5. A negative amount is rejected by the setter
        try:
            t.amount = -5
        except ValueError as error:
            print("Rejected:", error)

        # 6. The ID has no setter, so it can't be changed
        try:
            t.transaction_id = 99
        except AttributeError as error:
            print("Cannot change ID:", error)
