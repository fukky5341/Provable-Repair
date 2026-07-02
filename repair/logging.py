

def logging(message, option="a", line_break=1, border_type='-', border=None, logfile=None):
    print(message)

    if logfile is None:
        return

    line_break_str = "\n" * line_break
    
    border_top = None
    border_bottom = None
    if border == "top":
        str_len = len(message)
        border_top = border_type * str_len
    if border == "bottom":
        str_len = len(message)
        border_bottom = border_type * str_len
    elif border == "both":
        str_len = len(message)
        border_top = border_type * str_len
        border_bottom = border_type * str_len

    with open(logfile, option) as f:
        f.write(line_break_str)
        if border_top is not None:
            f.write(border_top + "\n")
        f.write(message + "\n")
        if border_bottom is not None:
            f.write(border_bottom + "\n")