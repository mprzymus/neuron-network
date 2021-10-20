import statistics

ACC = 1000


def show_history(history, label):
    string_list = ''
    for value in history:
        string_list += f'{value} & '
    print(
        f"{label} & {string_list} {round(statistics.mean(history) * 10) / 10} & {round(statistics.stdev(history) * 10) / 10}\\\\")


def show_stats(metrics, label):
    string_list = ''
    for value in metrics:
        string_list += f'{value} & '
    print(
        f"{label} & {string_list} & {round(metrics[0] / (metrics[0] + metrics[2]) * ACC) / ACC}& {round(metrics[1] / (metrics[1] + metrics[3]) * ACC) / ACC}\\\\")
