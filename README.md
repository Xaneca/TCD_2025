# TCD_2025
Cadeira de TCD. Mestrado em Inteligência Artificial e Ciência de Dados

# META 1 - Preparação de Dados
- [ ] 1. Criar script
- [ ] 2. DataSet
- [ ]      2.1 CArregar dados indivíduo -> NumPy
- [ ] 3. Outliers - 3-5% outliers - identificar e tratar
        a)   Univariável
        b)   Multivariável
        c)   Cálcular módulos dos vectores de aceleração, giroscópio, magnetómetro -> vt = (tx, ty, tz)
        d)   || vt || =  √( (tx)^2 + (ty)^2 + (tz)^2 )
- [ ]      3.1. BoxPlot de cada atividade - todos os sujeitos
- [ ]             coluna 12 - eixo horizontal
- [ ]             Eixo vertical - modulo dos vetores (1/grafico)
- [ ]             MatPlotLib
- [ ]      3.2. Densidade Outliers - olhar para os módulos dos vetores
- [ ]             fórmula: d = (n0)/(nr) * 100
- [ ]               n0: nº pontos classificados como outliers
- [ ]               nr: nº total pontos
- [ ]      3.3. Função: parametro Array de amostrar de uma variável - identificar outliers usando Z-score, para um K variável
- [ ]      3.4. Com o Z-score criado - assinalar os outliers - nos modulos de aceleraçao, giroscopio


# Meta 2 - Aprendizagem Computacional & Avaliação

