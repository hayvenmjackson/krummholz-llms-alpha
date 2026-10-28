"""Deterministic calculator for the read-only tool broker (tool `calculate`).

Arithmetic, unit conversion and date arithmetic go to code, never to the model's memory.
The expression is parsed with `ast` and evaluated over a closed set of node types and
functions; names, attributes, subscripts, comprehensions and calls to anything not listed
are rejected. Sizes are bounded so a proposal cannot exhaust memory or time.
"""
import ast, datetime as dt, math, operator

MAX_CHARS = 300
MAX_ABS = 1e15
MAX_POW = 64

UNITS = {  # factor to the base unit of each dimension
    "length": {"meters": 1.0, "kilometers": 1000.0, "centimeters": 0.01, "millimeters": 0.001, "miles": 1609.344,
               "feet": 0.3048, "inches": 0.0254, "yards": 0.9144, "nautical miles": 1852.0},
    "mass": {"kilograms": 1.0, "grams": 0.001, "pounds": 0.45359237, "ounces": 0.028349523125, "tonnes": 1000.0,
             "short tons": 907.18474},
    "area": {"square meters": 1.0, "hectares": 10000.0, "acres": 4046.8564224, "square feet": 0.09290304,
             "square kilometers": 1e6, "square miles": 2589988.110336},
    "volume": {"liters": 1.0, "milliliters": 0.001, "gallons": 3.785411784, "quarts": 0.946352946,
               "cubic meters": 1000.0, "cubic feet": 28.316846592, "board feet": 2.359737216},
    "speed": {"meters per second": 1.0, "kilometers per hour": 1 / 3.6, "miles per hour": 0.44704, "knots": 1852 / 3600},
}
ALIASES = {"meter": "meters", "m": "meters", "km": "kilometers", "kilometer": "kilometers", "cm": "centimeters",
           "centimeter": "centimeters", "mm": "millimeters", "mile": "miles", "mi": "miles", "foot": "feet", "ft": "feet",
           "inch": "inches", "in": "inches", "yard": "yards", "yd": "yards", "kilogram": "kilograms", "kg": "kilograms",
           "gram": "grams", "g": "grams", "pound": "pounds", "lb": "pounds", "lbs": "pounds", "ounce": "ounces",
           "oz": "ounces", "tonne": "tonnes", "metric tons": "tonnes", "hectare": "hectares", "ha": "hectares",
           "acre": "acres", "liter": "liters", "l": "liters", "litre": "liters", "litres": "liters", "gallon": "gallons",
           "gal": "gallons", "kph": "kilometers per hour", "km/h": "kilometers per hour", "mph": "miles per hour",
           "f": "fahrenheit", "c": "celsius", "k": "kelvin"}
TEMPERATURE = {"fahrenheit", "celsius", "kelvin"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class CalcError(ValueError):
    pass


def _unit(name):
    if type(name) is not str:
        raise CalcError("unit_must_be_text")
    u = " ".join(name.lower().split())
    return ALIASES.get(u, u)


def convert(value, source, target):
    s, t = _unit(source), _unit(target)
    if s in TEMPERATURE or t in TEMPERATURE:
        if not (s in TEMPERATURE and t in TEMPERATURE):
            raise CalcError("incompatible_units")
        c = {"celsius": value, "fahrenheit": (value - 32) * 5 / 9, "kelvin": value - 273.15}[s]
        return {"celsius": c, "fahrenheit": c * 9 / 5 + 32, "kelvin": c + 273.15}[t]
    for table in UNITS.values():
        if s in table and t in table:
            return value * table[s] / table[t]
    raise CalcError("unknown_or_incompatible_units")


def _date(text):
    try:
        return dt.date.fromisoformat(text)
    except (TypeError, ValueError):
        raise CalcError("dates_must_be_iso_yyyy_mm_dd") from None


def days_between(start, end):
    return (_date(end) - _date(start)).days


def weekday(date):
    return WEEKDAYS[_date(date).weekday()]


def add_days(date, days):
    if type(days) is not int:
        raise CalcError("days_must_be_integer")
    try:
        return (_date(date) + dt.timedelta(days=days)).isoformat()
    except OverflowError:
        raise CalcError("date_out_of_range") from None


def count(text, part):
    """Occurrences of `part` in `text`, ignoring case ("how many r's in strawberry")."""
    if type(text) is not str or type(part) is not str or not part:
        raise CalcError("count_needs_text_and_part")
    t, p = text.lower(), part.lower()
    return sum(1 for i in range(len(t) - len(p) + 1) if t[i:i + len(p)] == p)  # overlapping, like a person counts


def length(text):
    """Letters in a word or phrase (spaces and punctuation are not letters)."""
    if type(text) is not str:
        raise CalcError("length_needs_text")
    return sum(ch.isalpha() for ch in text)


def spell(text):
    """The letters of a word, separated, for spelling and position questions."""
    if type(text) is not str:
        raise CalcError("spell_needs_text")
    return "-".join(ch for ch in text if not ch.isspace())


def compare(a, b):
    """Which of two numbers is larger, as text ("9.9 > 9.11")."""
    if isinstance(a, str) or isinstance(b, str):
        raise CalcError("compare_needs_numbers")
    sign = ">" if a > b else "<" if a < b else "="
    return f"{display(a)} {sign} {display(b)}"


FUNCTIONS = {"convert": convert, "days_between": days_between, "weekday": weekday, "add_days": add_days,
             "count": count, "length": length, "spell": spell, "compare": compare,
             "sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log": math.log, "log10": math.log10,
             "exp": math.exp}
CONSTANTS = {"pi": math.pi, "e": math.e}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _check(v):
    if isinstance(v, bool):
        raise CalcError("unsupported_value")
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v) or abs(v) > MAX_ABS:
            raise CalcError("result_out_of_range")
    elif not isinstance(v, str):
        raise CalcError("unsupported_value")
    return v


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, str)):
            raise CalcError("unsupported_constant")
        if isinstance(node.value, str) and len(node.value) > 40:
            raise CalcError("text_too_long")
        return _check(node.value)
    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalcError("unknown_name")
        return CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(a, str) or isinstance(b, str):
            raise CalcError("arithmetic_on_text")
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_POW:
            raise CalcError("exponent_too_large")
        try:
            return _check(BINARY[type(node.op)](a, b))
        except (ZeroDivisionError, OverflowError):
            raise CalcError("undefined_result") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        v = _eval(node.operand)
        if isinstance(v, str):
            raise CalcError("arithmetic_on_text")
        return _check(UNARY[type(node.op)](v))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        if len(node.args) > 4:
            raise CalcError("too_many_arguments")
        args = [_eval(a) for a in node.args]
        try:
            return _check(FUNCTIONS[node.func.id](*args))
        except CalcError:
            raise
        except (TypeError, ValueError, OverflowError):
            raise CalcError("invalid_arguments") from None
    raise CalcError("unsupported_expression")


def evaluate(expression):
    """Return the value of a calculator expression or raise CalcError with a stable reason."""
    if type(expression) is not str or not expression.strip() or len(expression) > MAX_CHARS:
        raise CalcError("invalid_expression")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalcError("syntax_error") from None
    if sum(1 for _ in ast.walk(tree)) > 120:
        raise CalcError("expression_too_complex")
    return _eval(tree)


def display(value):
    """Stable text form: integers stay integers, floats get at most 6 significant decimals."""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e12:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) >= 1e6 or abs(value) < 1e-3 else f"{round(value, 6):,.6f}".rstrip("0").rstrip(".").replace(",", "")
    return str(value)


def run(arguments):
    """Broker entry: {"expression": str} -> calculator response (never raises)."""
    try:
        if type(arguments) is not dict or set(arguments) != {"expression"}:
            raise CalcError("invalid_arguments")
        value = evaluate(arguments["expression"])
        return {"status": "calculated", "expression": arguments["expression"], "result": display(value), "exact_code": True}
    except CalcError as error:
        return {"status": "unresolved", "reason": str(error)}
