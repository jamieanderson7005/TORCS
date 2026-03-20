from prac import get_weather, add, divide, UserManager
import pytest

# def test_get_weather():
#     assert get_weather(21) == 'cold'

# def test_add():
#     assert add(2,3) == 5, "2+3 should be 5"
#     assert add(-1,1) == 0, "-1+1 should be 0"
#     assert add(0,0) == 0, "0+0 should be 0"

# def test_divide():
#     with pytest.raises(ValueError, match="Cannot divide by zero"):
#         divide(10,0)

@pytest.fixture
def user_manager():
    """Creates a fresh instance of UserManager before each test"""
    return UserManager()

def test_add_user(user_manager):
    assert user_manager.add_user('john_doe', 'john@example.com') == True
    assert user_manager.get_user('john_doe') =='john@example.com'

def test_add_duplicate_user(user_manager):
    user_manager.add_user('john_doe', 'john@example.com')
    with pytest.raises(ValueError):
        user_manager.add_user('john_doe', 'another@example.com')