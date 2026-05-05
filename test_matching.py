from matching import Matrix, hungarian_match


def test_hungarian_basic_minimize():
    # Cost matrix where optimal assignment is (0->0,1->1)
    m = Matrix(2, 2)
    m.set(0, 0, 1)
    m.set(0, 1, 5)
    m.set(1, 0, 4)
    m.set(1, 1, 2)

    assignment = hungarian_match(m, maximize=False)
    return set(assignment) == {(0, 0), (1, 1)}


def test_hungarian_basic_maximize():
    # Score matrix where optimal assignment is (0->1,1->0)
    m = Matrix(2, 2)
    m.set(0, 0, 1)
    m.set(0, 1, 10)
    m.set(1, 0, 9)
    m.set(1, 1, 2)

    assignment = hungarian_match(m, maximize=True)
    return set(assignment) == {(0, 1), (1, 0)}


def _run():
    tests = [
        ("test_hungarian_basic_minimize", test_hungarian_basic_minimize),
        ("test_hungarian_basic_maximize", test_hungarian_basic_maximize),
    ]

    for name, fn in tests:
        try:
            ok = fn()
            print(f"{name}: {'PASS' if ok else 'FAIL'}")
        except ImportError as e:
            print(f"{name}: SKIP (missing dependency: {e})")
        except Exception as e:
            print(f"{name}: ERROR ({e})")


if __name__ == '__main__':
    _run()
