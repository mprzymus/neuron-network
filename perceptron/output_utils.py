import statistics


def show_history(history, label):
    string_list = ''
    for value in history:
        string_list += f'{value} & '
    print(
        f"{label} & {string_list} {round(statistics.mean(history) * 10) / 10} & {round(statistics.stdev(history) * 10) / 10}\\\\")