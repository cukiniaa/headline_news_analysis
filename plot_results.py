import pandas as pd
from bokeh.plotting import figure, show
from bokeh.palettes import Category20, Category20b
from bokeh.models import ColumnDataSource, Legend

filename = 'data/classified-abcnews-date-text.csv'
df = pd.read_csv(filename)

df['category'] = df['category'].astype('category')
df['publish_date'] = pd.to_datetime(df['publish_date'], format="%Y%m%d")
df['publish_date'] = df['publish_date'].dt.to_period('Y').dt.to_timestamp()

print(df.info())
print(df.head())

grouped = df.groupby(by=['publish_date', 'category'],
                     as_index=False).agg(count=('headline_text', 'count'))
grouped['category'] = grouped['category'].astype('category')

cat_ind = df['category'].cat.categories
colors = Category20[20]
if len(cat_ind) > len(colors):
    colors = colors + Category20b[20][:(len(cat_ind) - len(colors))]

# if we have to assign a color to every row:
# grouped['color'] = [colors[cat_ind.get_loc(cat)] for cat in grouped['category']]

fig = figure(plot_width=800, plot_height=800, x_axis_type='datetime')
fig.add_layout(Legend(), 'right')

X = []
Y = []
categories = grouped['category'].unique()
for category in categories:
    filtered = grouped[grouped['category'] == category]
    X.append(list(filtered['publish_date']))
    Y.append(list(filtered['count']))

source = ColumnDataSource(dict(
    xs=X,
    ys=Y,
    color=colors,
    category=categories
))

fig.multi_line(source=source, xs='xs', ys='ys', legend_group='category',
               color='color', line_width=2)

# TODO interactive legend
show(fig)
