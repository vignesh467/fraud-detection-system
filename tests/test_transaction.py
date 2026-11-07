import pytest

from src.transaction import Transaction


def make_transaction(amount=100.0):
    features = {f"V{i}": 0.0 for i in range(1, 29)}
    return Transaction("t1", amount, 10.0, features)


def test_properties_return_values():
    t = make_transaction(250.0)
    assert t.transaction_id == "t1"
    assert t.amount == 250.0
    assert t.time == 10.0


def test_negative_amount_is_rejected():
    with pytest.raises(ValueError):
        make_transaction(-5)


def test_setter_also_validates():
    t = make_transaction()
    with pytest.raises(ValueError):
        t.amount = -1


def test_private_attribute_is_hidden():
    t = make_transaction()
    with pytest.raises(AttributeError):
        _ = t.__amount


def test_to_dict_has_dataset_column_order():
    columns = list(make_transaction().to_dict().keys())
    assert columns[0] == "Time"
    assert columns[-1] == "Amount"
    assert len(columns) == 30
