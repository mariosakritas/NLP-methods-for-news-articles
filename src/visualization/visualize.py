
def plot_signals(mixed_df, fig, ax, keyword = 'keyword'):
    fig.autofmt_xdate(rotation=75)
    ax.plot(mixed_df.index, mixed_df.dw, color = 'k')
    # ax.set_xticks(dw_mentions.index.astype(str)[::8])
    ax.set_xlabel('Time', fontsize = 20)
    ax.set_ylabel(f'DW Articles per week \nwith {keyword} in keywords', color = 'k', fontsize = 16)
    ax2 = ax.twinx()
    ax2.plot(mixed_df.index, mixed_df.google, color = 'r', alpha =0.5) #may need to add .astype(str)
    ax2.set_ylabel(f'Relative amount of \nGoogle Searches for {keyword}', color = 'r', fontsize = 16)
    mixed_df.google = mixed_df.google.apply(lambda x: x-mixed_df.google.mean())
    mixed_df.dw = mixed_df.dw.apply(lambda x: x-mixed_df.dw.mean())

#count amount of predictions for each category
def count_values_for_columns(result):
    val_counts_for_cols = {}
    for col in result.columns:
        val_counts = result.explode(col)[col].value_counts()
        val_counts_for_cols[col] = val_counts
    return val_counts_for_cols