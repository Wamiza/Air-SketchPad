from canvas import Canvas


def test_draw_then_erase_same_spot_removes_ink():
    c = Canvas(640, 480)
    c.apply([("draw", (0.5, 0.5)), ("draw", (0.55, 0.5)), ("stroke_end", None)])
    assert c.has_ink()
    c.apply([("erase", (0.5, 0.5)), ("erase", (0.55, 0.5))])
    assert not c.has_ink()


def test_strokes_are_connected_and_end_cleanly():
    c = Canvas(640, 480)
    c.apply([("draw", (0.1, 0.5)), ("draw", (0.9, 0.5))])
    row = c.img[240]
    assert row[100:500].any(axis=1).all()          # a continuous line between the two points
    c.apply([("stroke_end", None), ("draw", (0.5, 0.1))])
    # no line is drawn between the end of the old stroke (0.9, 0.5) and the new dot (0.5, 0.1)
    assert not c.img[130:160, 430:470].any()


def test_colour_cycles_and_clear_wipes():
    c = Canvas(320, 240)
    first = c.colour_name
    c.apply([("colour", None)])
    assert c.colour_name != first
    for _ in range(len(Canvas.COLOURS) - 1):
        c.apply([("colour", None)])
    assert c.colour_name == first
    c.apply([("draw", (0.5, 0.5)), ("clear", None)])
    assert not c.has_ink()
