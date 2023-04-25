from __future__ import annotations

import july
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def mode(s: pd.Series) -> int | None:
    m = s.mode().values
    if len(m) == 1:
        r, = m
        return r
    else:
        return None


def main() -> int:
    df = pd.read_csv(
        'data/classified_rgs.csv',
        index_col='date',
        parse_dates=['date'],
    )
    print(f'total number of images: {len(df)}')
    print(f'data between: {df.index.min()} and {df.index.max()}')

    df_daily = df.resample('1D').agg({
        'cloud_class': mode,
        'cloud_frac': 'mean',
    })
    df_hourly = df.resample('60min').agg({
        'cloud_class': mode,
        'cloud_frac': 'mean',
    })

    _, ax = plt.subplots(figsize=(16, 12))
    july.heatmap(
        dates=df_daily.index,
        data=df_daily['cloud_frac'],
        cmap='viridis',
        colorbar=True,
        value_label=True,
        ax=ax,
        horizontal=True,
        month_grid=True,
        fontfamily='Sans-serif',
        fontsize=7,
    )
    plt.savefig('figs/cal_plot.png', dpi=600)
    plt.savefig('figs/cal_plot.svg')
    plt.close()

    grouped = df.groupby('cloud_class', as_index=False).count()
    grouped['cloud_frac'] = grouped['cloud_frac'] * 100 / len(df_hourly)

    colors = sns.color_palette('colorblind')[0:5]

    plt.pie(
        x=grouped['cloud_frac'],
        labels=grouped['cloud_class'],
        colors=colors,
        autopct='%.0f%%',
    )
    plt.savefig('figs/pie.png', dpi=600)
    plt.savefig('figs/pie.svg')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
