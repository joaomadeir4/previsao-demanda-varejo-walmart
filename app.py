import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dataviz - Walmart M5", layout="wide")

st.markdown("""
<style>
  body, .stApp { background-color: #0D1117; }
  .block-container { padding-top: 2rem; max-width: 1100px; }
  .chapter { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; color: #7C5CFC; margin-bottom: 0.2rem; }
  .section-q { font-size: 1.35rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; line-height: 1.3; }
  .section-context { font-size: 0.9rem; color: #64748b; margin-bottom: 1.5rem; line-height: 1.75; max-width: 760px; }
  .insight { background: #111827; border-left: 3px solid #7C5CFC; padding: 0.9rem 1.2rem; border-radius: 0 6px 6px 0; margin: 1rem 0 2rem 0; font-size: 0.875rem; color: #94a3b8; line-height: 1.75; }
  .insight strong { color: #a78bfa; }
  .insight .takeaway { display: block; margin-top: 0.5rem; color: #e2e8f0; font-weight: 600; font-size: 0.9rem; }
  .connector { font-size: 0.85rem; color: #475569; font-style: italic; margin: 2rem 0 1.5rem 0; padding-left: 1rem; border-left: 1px solid #1f2d45; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font_color='#94a3b8', title_font_color='#e2e8f0', title_font_size=14,
    margin=dict(t=50, b=30, l=10, r=10),
)
GRID = dict(gridcolor='#1f2d45', zerolinecolor='#1f2d45')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "silver")


@st.cache_data
def load_aggs():
    agg_daily = pd.read_parquet(os.path.join(DATA_DIR, "agg_daily.parquet"), engine="pyarrow")
    agg_daily["date"] = pd.to_datetime(agg_daily["date"])
    agg_daily_store = pd.read_parquet(os.path.join(DATA_DIR, "agg_daily_store.parquet"), engine="pyarrow")
    agg_daily_store["date"] = pd.to_datetime(agg_daily_store["date"])
    agg_daily_cat = pd.read_parquet(os.path.join(DATA_DIR, "agg_daily_cat.parquet"), engine="pyarrow")
    agg_daily_cat["date"] = pd.to_datetime(agg_daily_cat["date"])
    agg_store_dept = pd.read_parquet(os.path.join(DATA_DIR, "agg_store_dept.parquet"), engine="pyarrow")
    agg_store_item = pd.read_parquet(os.path.join(DATA_DIR, "agg_store_item.parquet"), engine="pyarrow")
    return agg_daily, agg_daily_store, agg_daily_cat, agg_store_dept, agg_store_item


agg_daily, agg_daily_store, agg_daily_cat, agg_store_dept, agg_store_item = load_aggs()

st.markdown("## Retail Predictive Forecasting — Walmart M5")
st.markdown('<div class="section-context">Análise de demanda do dado bruto até o sinal que o modelo de previsão precisa captar. Cada bloco parte de uma pergunta de negócio antes de entrar no gráfico.</div>', unsafe_allow_html=True)
st.divider()

# 1. VISÃO MACRO
st.markdown('<div class="chapter">CAPÍTULO 1 · VISÃO MACRO</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">Qual é o tamanho e o ritmo dessa operação?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Antes de treinar qualquer modelo, precisa entender a base histórica: a inércia da demanda, onde estão os picos fora da curva e a tendência de longo prazo que o algoritmo vai ter que aprender.</div>', unsafe_allow_html=True)

trend_global = agg_daily[['date', 'revenue']].copy()
trend_global['media_movel_30d'] = trend_global['revenue'].rolling(30, center=True).mean()
receita_total = trend_global['revenue'].sum()
receita_media = trend_global['revenue'].mean()
pico_data = trend_global.loc[trend_global['revenue'].idxmax(), 'date'].strftime('%b %Y')
pico_val = trend_global['revenue'].max()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=trend_global['date'], y=trend_global['revenue'],
    name='Faturamento Diário', line=dict(color='#334155', width=1), opacity=0.6))
fig1.add_trace(go.Scatter(x=trend_global['date'], y=trend_global['media_movel_30d'],
    name='Média Móvel 30d', line=dict(color='#7C5CFC', width=2.5)))
fig1.update_layout(**PLOT_LAYOUT,
    title='Faturamento diário mostra crescimento consistente com sazonalidade anual clara',
    yaxis=dict(title='Faturamento (R$)', **GRID), xaxis=dict(showgrid=False),
    legend=dict(orientation='h', y=1.08, x=0))
st.plotly_chart(fig1, use_container_width=True)
st.markdown(f'<div class="insight">A operação acumula <strong>R$ {receita_total:,.0f}</strong> no período, com média diária de <strong>R$ {receita_media:,.0f}</strong>. O pico histórico foi em <strong>{pico_data}</strong>, R$ {pico_val:,.0f} num único dia.<span class="takeaway">A tendência de fundo é de crescimento e os picos sazonais se repetem ano a ano, ou seja, são previsíveis.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">Dá pra dimensionar a operação. Falta entender o comportamento de compra dentro da semana.</div>', unsafe_allow_html=True)
st.divider()

# 2. SAZONALIDADE SEMANAL
st.markdown('<div class="chapter">CAPÍTULO 2 · COMPORTAMENTO SEMANAL</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">O dia da semana importa para o modelo?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Sem variáveis de calendário, o algoritmo fica cego para o principal driver de tráfego nas lojas. Vale medir o peso real de cada dia antes de decidir quais features entram.</div>', unsafe_allow_html=True)

traducao = {'Monday':'Segunda','Tuesday':'Terça','Wednesday':'Quarta',
            'Thursday':'Quinta','Friday':'Sexta','Saturday':'Sábado','Sunday':'Domingo'}
ordem_pt = ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo']
trend_dia = trend_global[['date', 'revenue']].copy()
trend_dia['dia'] = trend_dia['date'].dt.day_name().map(traducao)
media_dia = trend_dia.groupby('dia')['revenue'].mean().reindex(ordem_pt).reset_index()
media_global_dia = media_dia['revenue'].mean()
media_dia['variacao'] = (media_dia['revenue'] / media_global_dia - 1) * 100
media_dia['destaque'] = media_dia['variacao'] > 5

fig2 = go.Figure()
for _, row in media_dia.iterrows():
    fig2.add_trace(go.Bar(x=[row['dia']], y=[row['revenue']],
        marker_color='#7C5CFC' if row['destaque'] else '#1e2d45',
        showlegend=False, text=f"{row['variacao']:+.1f}%", textposition='outside',
        textfont=dict(color='#a78bfa' if row['destaque'] else '#475569', size=11)))
fig2.add_hline(y=media_global_dia, line_dash='dash', line_color='#475569',
    annotation_text='Média', annotation_font_color='#475569', annotation_position='right')
fig2.update_layout(**PLOT_LAYOUT,
    title='Fim de semana concentra a demanda — sábado e domingo lideram com folga',
    yaxis=dict(title='Receita Média Diária (R$)', **GRID, showgrid=False),
    xaxis=dict(showgrid=False, categoryorder='array', categoryarray=ordem_pt), bargap=0.35)
st.plotly_chart(fig2, use_container_width=True)

dia_pico = media_dia.loc[media_dia['revenue'].idxmax(), 'dia']
var_pico = media_dia.loc[media_dia['revenue'].idxmax(), 'variacao']
st.markdown(f'<div class="insight">O pico é no <strong>{dia_pico}</strong>, {var_pico:.1f}% acima da média semanal. Segunda-feira registra o menor volume, formando um vale que se repete toda semana.<span class="takeaway">Dia da semana precisa entrar como feature. Sem isso o modelo tende a errar justamente nos extremos.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">O padrão semanal está claro. O que tira essa sazonalidade do previsível são os eventos fora do calendário comum.</div>', unsafe_allow_html=True)
st.divider()

# 3. FERIADOS
st.markdown('<div class="chapter">CAPÍTULO 3 · CHOQUES EXTERNOS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">Quais eventos injetam ou drenam receita fora do padrão?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Feriados e eventos especiais não são ruído, são sinal. O modelo precisa saber diferenciar um Super Bowl de uma segunda-feira qualquer.</div>', unsafe_allow_html=True)

if 'event_name_1' in agg_daily.columns:
    trend_g = agg_daily[['date', 'revenue']]
    events = agg_daily[agg_daily['event_name_1'].notna()][['date', 'event_name_1']].drop_duplicates()
    events = events[events['event_name_1'] != 'nan']
    holiday = trend_g.merge(events, on='date', how='inner')
    media_normal = trend_g[~trend_g['date'].isin(events['date'])]['revenue'].mean()
    impacto = holiday.groupby('event_name_1', observed=True)['revenue'].mean().reset_index()
    impacto['variacao'] = (impacto['revenue'] / media_normal - 1)
    impacto = impacto.sort_values('variacao')
    top = pd.concat([impacto.head(6), impacto.tail(6)]).drop_duplicates()
    cores = ['#f87171' if v < 0 else '#7C5CFC' for v in top['variacao']]
    fig3 = go.Figure(go.Bar(x=top['variacao'], y=top['event_name_1'], orientation='h',
        marker_color=cores, text=[f"{v:+.0%}" for v in top['variacao']],
        textposition='outside', textfont=dict(size=11, color='#94a3b8')))
    fig3.add_vline(x=0, line_color='#334155', line_width=1.5)
    fig3.update_layout(**PLOT_LAYOUT,
        title='Super Bowl impulsiona vendas; Natal fecha as lojas — ambos críticos para o modelo',
        xaxis=dict(tickformat='.0%', showgrid=False, title='Variação vs. dia normal'),
        yaxis=dict(showgrid=False))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('<div class="insight">Do lado positivo, Super Bowl e Labor Day puxam a demanda pra cima. Já o Natal derruba as vendas quase a zero, confirmando o fechamento das lojas.<span class="takeaway">Ignorar esses eventos gera erro sistemático em datas conhecidas com antecedência. Vale tratar como feature, no mesmo nível de dia da semana.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">O padrão temporal está mapeado. Falta ver se toda loja se comporta do mesmo jeito.</div>', unsafe_allow_html=True)
st.divider()

# 4. LOJAS E ESTADOS
st.markdown('<div class="chapter">CAPÍTULO 4 · LOJAS E ESTADOS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">A localização importa ou todas as lojas são parecidas?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Se a disparidade entre lojas for alta, o modelo vai precisar de variável categórica de localização. Se for baixa, um modelo global já resolve.</div>', unsafe_allow_html=True)

store_perf = agg_daily_store.groupby(['store_id', 'state_id'], observed=True).agg(
    dias=('date', 'nunique'), faturamento=('revenue', 'sum')
).reset_index()
store_perf['media_diaria'] = store_perf['faturamento'] / store_perf['dias']
media_global = store_perf['media_diaria'].mean()
store_perf = store_perf.sort_values('media_diaria', ascending=True)
cores_estado = {'CA':'#7C5CFC','TX':'#a78bfa','WI':'#4ade80'}

fig4 = go.Figure()
for estado in store_perf['state_id'].unique():
    sub = store_perf[store_perf['state_id'] == estado]
    fig4.add_trace(go.Bar(x=sub['media_diaria'], y=sub['store_id'], orientation='h',
        name=estado, marker_color=cores_estado.get(estado, '#334155')))
fig4.add_vline(x=media_global, line_dash='dash', line_color='#f59e0b',
    annotation_text=f'Média: R${media_global:,.0f}', annotation_font_color='#f59e0b',
    annotation_position='top right')
fig4.update_layout(**PLOT_LAYOUT,
    title='Lojas da Califórnia lideram; variação entre unidades justifica variável de localização',
    xaxis=dict(title='Faturamento Médio Diário (R$)', showgrid=False),
    yaxis=dict(showgrid=False), legend=dict(title='Estado', orientation='h', y=1.08))
st.plotly_chart(fig4, use_container_width=True)

loja_top = store_perf.iloc[-1]
loja_bot = store_perf.iloc[0]
ratio = loja_top['media_diaria'] / loja_bot['media_diaria']
st.markdown(f'<div class="insight">A loja de maior faturamento ({loja_top["store_id"]}) vende {ratio:.1f}x mais por dia que a menor ({loja_bot["store_id"]}).<span class="takeaway">Essa dispersão é grande demais pra ignorar: o modelo precisa de variável de loja ou estado, senão vai subestimar as unidades de maior volume.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">Sabemos onde o dinheiro entra. O próximo ponto é o que essas lojas vendem.</div>', unsafe_allow_html=True)
st.divider()

# 5. CATEGORIAS
st.markdown('<div class="chapter">CAPÍTULO 5 · CATEGORIAS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">Quais categorias sustentam o faturamento e qual é a mais estável?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Entender a composição da receita por categoria ajuda a decidir onde o modelo precisa ser mais preciso. Categoria com sazonalidade forte pede features temporais mais ricas.</div>', unsafe_allow_html=True)

cat_trend = agg_daily_cat
cat_total = agg_daily_cat.groupby('cat_id', observed=True)['revenue'].sum().reset_index()
cat_total['share'] = cat_total['revenue'] / cat_total['revenue'].sum() * 100
cat_dom = cat_total.loc[cat_total['revenue'].idxmax(), 'cat_id']
cat_dom_share = cat_total.loc[cat_total['revenue'].idxmax(), 'share']

fig5 = px.line(cat_trend, x='date', y='revenue', color='cat_id',
    title=f'{cat_dom} domina a receita — as demais categorias são secundárias e mais estáveis',
    color_discrete_sequence=['#7C5CFC','#a78bfa','#4ade80'])
fig5.update_layout(**PLOT_LAYOUT,
    yaxis=dict(title='Faturamento (R$)', **GRID), xaxis=dict(showgrid=False),
    legend=dict(title='Categoria', orientation='h', y=1.08))
st.plotly_chart(fig5, use_container_width=True)
st.markdown(f'<div class="insight">{cat_dom} responde por {cat_dom_share:.1f}% de toda a receita. HOBBIES e HOUSEHOLD têm volume menor, mas comportamento bem mais estável.<span class="takeaway">FOODS vai precisar de features de calendário mais ricas. Tratar as três categorias do mesmo jeito seria erro.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">FOODS domina em volume. Falta ver se, dentro dela, todo produto pesa igual.</div>', unsafe_allow_html=True)
st.divider()

# FILTRO
st.markdown('<div class="chapter">ANÁLISE POR LOJA — CAPÍTULOS 6, 7 E 8</div>', unsafe_allow_html=True)
st.markdown('<div class="section-q">Selecione uma loja para descer ao nível micro</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Os blocos seguintes descem ao nível de departamento e SKU. Como cada loja tem seu próprio mix de produtos, o filtro é necessário para manter a análise precisa.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    estado = st.selectbox("Estado", sorted(agg_daily_store['state_id'].unique()))
with col2:
    loja = st.selectbox("Loja", sorted(agg_daily_store[agg_daily_store['state_id'] == estado]['store_id'].unique()))

st.divider()

# 6. DEPARTAMENTOS
st.markdown('<div class="chapter">CAPÍTULO 6 · DEPARTAMENTOS</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-q">Dentro de {loja}, qual departamento concentra o risco do modelo?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Já sabemos que FOODS domina no geral. Aqui a ideia é achar qual subdepartamento concentra o maior volume financeiro — e, por consequência, o maior risco de erro de previsão.</div>', unsafe_allow_html=True)

dept = agg_store_dept[agg_store_dept['store_id'] == loja][['dept_id', 'revenue']].copy()
dept['share'] = dept['revenue'] / dept['revenue'].sum() * 100
dept = dept.sort_values('revenue', ascending=True)
dept_top = dept.iloc[-1]
cores_dept = ['#7C5CFC' if d == dept_top['dept_id'] else '#1e2d45' for d in dept['dept_id']]

fig6 = go.Figure(go.Bar(x=dept['revenue'], y=dept['dept_id'], orientation='h',
    marker_color=cores_dept,
    text=[f"{s:.1f}%" for s in dept['share']], textposition='outside',
    textfont=dict(color='#64748b', size=11)))
fig6.update_layout(**PLOT_LAYOUT,
    title=f'{dept_top["dept_id"]} concentra {dept_top["share"]:.1f}% da receita de {loja} — departamento crítico',
    xaxis=dict(title='Faturamento Total (R$)', showgrid=False), yaxis=dict(showgrid=False))
st.plotly_chart(fig6, use_container_width=True)
st.markdown(f'<div class="insight">{dept_top["dept_id"]} responde por {dept_top["share"]:.1f}% da receita total de {loja}.<span class="takeaway">Um erro de 10% nesse departamento pesa mais no financeiro do que 50% de erro somado em todos os outros.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">O departamento crítico está identificado. Falta ver quais produtos dentro dele realmente importam.</div>', unsafe_allow_html=True)
st.divider()

# 7. CURVA ABC
st.markdown('<div class="chapter">CAPÍTULO 7 · CURVA ABC (PARETO)</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-q">Quais SKUs sustentam 80% do faturamento de {loja}?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">No nível mais baixo de granularidade, dá pra separar sinal de ruído. Os produtos Classe A precisam de previsão precisa, os demais toleram mais erro.</div>', unsafe_allow_html=True)

item_loja = agg_store_item[agg_store_item['store_id'] == loja]
pareto = item_loja[['item_id', 'revenue']].sort_values('revenue', ascending=False).reset_index(drop=True)
pareto['cumsum'] = pareto['revenue'].cumsum()
pareto['cumperc'] = pareto['cumsum'] / pareto['revenue'].sum() * 100
pareto['rank'] = range(1, len(pareto) + 1)
pareto['classe'] = pareto['cumperc'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
n_a = len(pareto[pareto['classe'] == 'A'])
pct_a = n_a / len(pareto) * 100
corte_a = pareto[pareto['classe'] == 'A']['rank'].max()

fig7 = go.Figure()
fig7.add_trace(go.Bar(x=pareto['rank'], y=pareto['revenue'],
    marker_color='#7C5CFC', opacity=0.5, name='Receita por SKU'))
fig7.add_trace(go.Scatter(x=pareto['rank'], y=pareto['cumperc'],
    yaxis='y2', name='% Acumulado', line=dict(color='#f59e0b', width=2.5)))
fig7.add_vline(x=corte_a, line_dash='dash', line_color='#4ade80',
    annotation_text=f'Classe A: {n_a} SKUs ({pct_a:.1f}%)',
    annotation_font_color='#4ade80', annotation_position='top right')
fig7.add_hline(y=80, line_dash='dot', line_color='#f59e0b', yref='y2',
    annotation_text='80%', annotation_font_color='#f59e0b', annotation_position='right')
fig7.update_layout(**PLOT_LAYOUT,
    title=f'{pct_a:.1f}% dos SKUs respondem por 80% da receita — concentração clássica de Pareto',
    yaxis=dict(title='Faturamento (R$)', showgrid=False),
    yaxis2=dict(title='% Acumulado', overlaying='y', side='right', range=[0,105], showgrid=False, ticksuffix='%'),
    xaxis=dict(title='Ranking de SKUs (maior → menor receita)', showgrid=False),
    legend=dict(orientation='h', y=1.08))
st.plotly_chart(fig7, use_container_width=True)
st.markdown(f'<div class="insight">Só {n_a} SKUs ({pct_a:.1f}% do portfólio de {loja}) respondem por 80% de toda a receita.<span class="takeaway">Faz mais sentido avaliar o modelo pelo erro nos Classe A. RMSE médio pra todos os SKUs mascara o que de fato importa financeiramente.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="connector">Sabemos quais produtos importam. Resta ver se eles vendem todo santo dia.</div>', unsafe_allow_html=True)
st.divider()

# 8. VENDA ZERO
st.markdown('<div class="chapter">CAPÍTULO 8 · INTERMITÊNCIA DE VENDAS</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-q">Quanto do silêncio nas prateleiras é padrão e não falha do modelo?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-context">Muitos produtos não vendem todo dia. Se o modelo não souber lidar com esse silêncio, vai interpretar zero como ausência de demanda quando na verdade é comportamento normal do SKU.</div>', unsafe_allow_html=True)

sparsity = item_loja[['item_id', 'zero_pct']].copy()
media_zeros = sparsity['zero_pct'].mean()
pct_mais50 = (sparsity['zero_pct'] > 50).mean() * 100

fig8 = go.Figure()
fig8.add_trace(go.Histogram(x=sparsity['zero_pct'], nbinsx=35,
    marker_color='#7C5CFC', opacity=0.75, name='SKUs'))
fig8.add_vline(x=media_zeros, line_dash='dash', line_color='#f87171',
    annotation_text=f'Média: {media_zeros:.1f}%', annotation_font_color='#f87171',
    annotation_position='top right')
fig8.add_vline(x=50, line_dash='dot', line_color='#f59e0b',
    annotation_text='50% zeros', annotation_font_color='#f59e0b',
    annotation_position='top left')
fig8.update_layout(**PLOT_LAYOUT,
    title=f'{pct_mais50:.1f}% dos SKUs ficam sem vender mais da metade dos dias — zeros são sinal, não ruído',
    xaxis=dict(title='% de dias com venda = 0', showgrid=False),
    yaxis=dict(title='Quantidade de SKUs', **GRID))
st.plotly_chart(fig8, use_container_width=True)
st.markdown(f'<div class="insight">Em média, cada SKU de {loja} passa {media_zeros:.1f}% dos dias sem venda, e {pct_mais50:.1f}% dos produtos ficam mais da metade do tempo zerados.<span class="takeaway">Agregar por semana ajuda o modelo a não se confundir: o silêncio de um dia isolado não pode contaminar a previsão do período todo.</span></div>', unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center; color:#334155; font-size:0.8rem; padding: 1rem 0 2rem 0;">Base <code>sales_slim.parquet</code> — pronta para Feature Engineering &nbsp;·&nbsp; João Madeira · Senior BI Analyst</div>', unsafe_allow_html=True)
