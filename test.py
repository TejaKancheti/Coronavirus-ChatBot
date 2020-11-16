
with open('abbrevations', 'r') as rd:
    file_lines = [','.join([x.strip(), str(x)]) for x in rd.readlines()]
with open('abbrevationsnnew', 'w') as f:
    f.writelines(file_lines)