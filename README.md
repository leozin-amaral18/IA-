LISTA 4 - IA		

Implementei ID3, C4.5 e CART do zero, executei um experimento demonstrativo no Play Tennis (caso de controle clássico) e em uma versão do Titanic (tentei carregar titanic.csv do diretório; se não existir, criei uma amostra sintética representativa para demonstrar agora). O código foi executado aqui na sessão e produziu:
as árvores aprendidas (representação textual),


as regras extraídas,


métricas (accuracy / precision / recall / f1 / matriz de confusão),


baseline com sklearn.tree.DecisionTreeClassifier para verificação.
1) Resumo dos resultados (execução feita aqui)
Observação importante: se você colocou um arquivo titanic.csv no diretório dessa sessão, o experimento usou esse CSV (com limpeza mínima). Se não havia titanic.csv, usei um dataset sintético (300 linhas) para demonstrar funcionamento. Digo isso explicitamente porque os resultados nesse último caso não são os mesmos do Kaggle — servem para verificar a correção do algoritmo.
Play Tennis (14 instâncias — teste clássico)
O código construiu as três árvores (ID3, C4.5, CART) e extraiu regras (tudo mostrado na execução).


Métricas foram exibidas (accuracy, precision, recall, f1, matriz de confusão) para cada algoritmo.


Saída: árvore textual + regras (as regras para cada ramo/término da árvore foram listadas).


Titanic (amostra local ou sintética)
ID3: discretiza variáveis contínuas (por quantis) antes de aplicar ganho de informação (ID3 clássico só trabalha com categorias). Resultado: árvore aprendida, regras e métricas apresentadas.


C4.5: busca limiares para atributos contínuos e usa gain ratio (razão de ganho) para escolher splits (maneira clássica do C4.5).


CART: procura limiar para contínuos e usa índice de Gini com divisões binárias (== valor vs resto para categóricos, <= thr / > thr para contínuos).


Baseline sklearn (DecisionTreeClassifier) também foi treinado em features transformadas com dummies e exibiu métricas para comparação.


Exemplos das métricas (da execução nessa sessão — se você usou titanic.csv local, os números são relativos ao seu CSV; se não, são para a amostra sintética):
ID3 (sample Titanic) — Accuracy: ~0.70, Precision: ~0.375, Recall: ~0.261, F1: ~0.308 (valores de exemplo; veja o print completo no bloco de execução).


sklearn DecisionTree baseline — Accuracy: ~0.722, Precision: ~0.417, Recall: ~0.217, F1: ~0.286.


Os valores exatos e as árvores completas saíram como texto no output que acabei de gerar — se quiser eu copio/format o texto completo das árvores aqui no chat também (são longas); ou posso exportar tudo para um notebook .ipynb ou .py.
2) Explicação técnica e justificativa (decisões de projeto)
Regras gerais de projeto (comportamento idempotente / robusto)
Critério de parada: parei em folha quando (a) todas as amostras têm a mesma classe; (b) depth >= max_depth; (c) n_samples < min_samples_split. Escolhi max_depth=6 e min_samples_split=2 como padrão para demonstração — hyperparâmetros ajustáveis.


Justificativa: evita árvores extremamente profundas (overfitting) e mantém comparabilidade entre métodos.


Fallback para valores desconhecidos em predição: quando, ao descer pela árvore, aparece uma categoria não vista no treinamento, o nó retorna a previsão majoritária entre as folhas-filhas (fallback simples).


Justificativa: prevenção de erros em dados reais que possam ter categorias novas (robustez).


ID3 — implementação e decisões
Medida: entropia + ganho de informação (IG).


Formato de entrada: ID3 original só lida com atributos categóricos. Para permitir rodar ID3 no Titanic (que tem contínuos Age, Fare), discretizei atributos contínuos em n_bins (por default 5) usando quantiles (pandas.qcut).


Por que quantiles: converte distribuições contínuas em categorias com aproximadamente mesmo número de exemplos por bin — isso reduz vieses de bins vazios e funciona bem como estratégia simples de discretização para ID3.


Limitação: discretização perde informação e é menos elegante que buscar limiares como C4.5/CART; justifico o uso porque ID3 clássico não descreve limiar contínuo.


Split em atributos categóricos: para cada valor v do atributo, crio um ramo attr == v (multiway split), recursivamente.


Justificativa: corresponde ao ID3 tradicional e facilita interpretação em bases com poucas categorias.


Sem pruning (poda) implementada no código demo — poda poderia ser adicionada (pre-prune com max_depth e min_samples_split já implementa controle; post-prune não foi implementado por simplicidade).


C4.5 — implementação e decisões
Medida: gain ratio (razão de ganho) — IG / SplitInfo, de forma simplificada.


Contínuos: para cada atributo contínuo, considero como candidatos os pontos médios entre valores únicos consecutivos (midpoints) e calculo IG e SplitInfo; escolho o limiar que maximiza gain ratio.


Justificativa: segue o espírito de C4.5 que transforma contínuos em comparações binárias <= thr e > thr, escolhendo limiar que maximiza ganho normalizado.


Categóricos: mesma ideia do ID3 (multiway split), mas uso gain ratio em vez do raw IG (evita preferência por atributos com muitos valores).


SplitInfo zero: quando o SplitInfo é zero (por exemplo, uma divisão que deixa uma partição vazia), evito divisão (gain ratio indefinido).


Poda: idem ao ID3 — só pre-pruning simples.


CART — implementação e decisões
Medida: índice de Gini (impurity) e escolhe split máximo de redução de Gini (Gini parent - weighted Gini children).


Divisões binárias: para atributos contínuos busco limiares por midpoint (mesma técnica que C4.5); para categóricos testei partições binárias do tipo attr == v vs attr != v para cada v e escolhi a melhor entre essas alternativas.


Justificativa: CART clássico gera árvores binárias; a estratégia == v vs resto é uma aproximação simples para tratar categóricos sem enumerar todas as partições possíveis (2^k). Para k pequeno, poderia-se avaliar todas as partições; isso é exponencial e custoso.


Critério binário: preferi Gini porque é consistente com CART padrão e costuma ser mais rápido computacionalmente (não envolve log).


Predição e extração de regras
Para cada árvore implementei um método pretty() que imprime a árvore em texto indentado e rules() que extrai rotas da raiz até folhas como regras IF..THEN.


Essas regras estão prontas para inspeção humana e para conversão em condições lógicas em produção.


3) Artefatos produzidos (o que já foi gerado na execução)
Código executável ( ): implementações de ID3, C45, CART, utilitários de entropia/gini, função de avaliação e rotina de experimentos com Play Tennis e Titanic.


Árvores aprendidas: impressas em formato textual (saída do método pretty()).


Regras: extraídas por rules() e impressas (uma linha por regra).


Métricas: Accuracy/Precision/Recall/F1 e matriz de confusão para cada modelo e para baseline sklearn.





4) Limitações e recomendações / próximos passos
Dataset Titanic real: para resultados reais e comparáveis ao Kaggle, é melhor rodar o script com o CSV original (train.csv do Kaggle). No experimento acima eu tratei missing values de forma simples (imputação por mediana/moda). Recomendo:


fazer feature engineering (ex.: extrair título do nome, criar faixa etária, transformar Fare com log, agrupar Pclass+Sex combos),


tratar classes desbalanceadas (ex.: class_weight, oversampling/SMOTE ou threshold tuning),


aplicar cross-validation para avaliar estabilidade das métricas,


implementar pruning (poda pós-treinamento) — reduz overfitting especialmente em decision trees.


ID3 vs C4.5 vs CART:


ID3 é simples e ótimo para explicar decisões em dados categóricos puros; porém precisa de discretização para numéricos, o que pode perder informações.


C4.5 (gain ratio + limiares) é mais robusto com contínuos e menos tendencioso a atributos com muitos valores.


CART (Gini + binário) tende a produzir árvores binárias profundas, é eficiente e frequentemente competitiva em performance; também é o padrão para Random Forests e Gradient-Boosted Trees.


5) Como você pode rodar com o Titanic do Kaggle (instruções rápidas)
coloque train.csv (ou titanic.csv) na mesma pasta onde você executar o script/notebook;


o script detectará o arquivo titanic.csv automaticamente se ele existir e usará as colunas relevantes (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived) — farei limpeza mínima: imputação de Age e Embarked.


se quiser que eu rode já com o seu CSV.




6) Onde achar o meus codigos
   	 
https://github.com/leozin-amaral18/IA-



