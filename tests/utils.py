import ast
import inspect


def base_class_constructor_checker(class_implementation, tester):
    tree = ast.parse(inspect.getsource(class_implementation))
    constructors = list(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    tester.assertLessEqual(len(constructors), 1, "Multiple constructors found")

    if len(constructors) == 0:
        # No constructor, base will be called by default
        return

    constructor = constructors[0]

    # Collect all function calls inside the constructor
    fn_calls = list(
        n
        for n in ast.walk(constructor)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    )

    # For each, check if there is something of the form super(...).__init(...)
    def is_base_constructor_call(node):
        fn_call = node.func
        if not isinstance(fn_call.value, ast.Call):
            return False
        if not fn_call.value.func.id == "super":
            return False
        if not fn_call.attr == "__init__":
            return False

        return True

    filtered_fn_calls = list(filter(is_base_constructor_call, fn_calls))

    tester.assertEqual(
        len(filtered_fn_calls), 1, "Call to base class constructor missing"
    )

    tester.assertTrue(
        any(
            isinstance(k.value, ast.Name) and k.value.id == "kwargs"
            for k in filtered_fn_calls[0].keywords
        ),
        "kwargs not passed to the base class constructor",
    )
