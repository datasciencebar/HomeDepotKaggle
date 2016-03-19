import unittest


class MyTestCase(unittest.TestCase):
    def test_fix_units(self):
        strings = [("Custom Building Products WonderBoard Lite 5 ft. x 3 ft. x 1/4 in. Backer Board",
                    "Custom Building Products WonderBoard Lite 5 ft x 3 ft x 1/4 in Backer Board"),
                   ("1/4 inch mel", "1/4 in mel"),
                   ("Sigman 6 in. Tarp Ball Bungee (25-Pack)", "Sigman 6 in Tarp Ball Bungee (25-Pack)"),
                   ("6 inches", "6 in"),
                   ("6' long", "6 in long"),
                   ("6'' long", "6 ft long"),
                   ("6in long", "6 in long"),
                   ("6-in.", "6 in"),
                   ("Pine Plywood (Common: 23/32 in. x 4 ft. x 8 ft.; Actual: 0.688 in. x 48 in. x 96 in.)",
                    "Pine Plywood (Common: 23/32 in x 4 ft x 8 ft ; Actual: 0.688 in x 48 in x 96 in )"),
                   ("3/4-in lumber", "3/4 in lumber"),
                   ("container 90''x12''", "container 90 ft x 12 ft"),
                   ("10ft.x7ft. garage door", "10 ft x 7 ft garage door"),
                   ("12ft COUNTER TOP", "12 ft COUNTER TOP"),
                   ("duct tape hvac 6 inch r8", "duct tape hvac 6 in r8")
                   ]
        from model import fix_units

        for test_string, correct_string in strings:
            fixed_string = fix_units(test_string)
            assert fixed_string == correct_string, fixed_string + " != " + correct_string

if __name__ == '__main__':
    unittest.main()
