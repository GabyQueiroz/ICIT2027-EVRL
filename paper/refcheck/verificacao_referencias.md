# Verificação das referências da introdução

Cada referência foi conferida no Crossref (DOI, autores, ano, veículo) e o resumo foi lido
(Semantic Scholar / OpenAlex; texto completo quando indicado). A coluna "Afirmação usada" é o
que o artigo pode dizer apoiado na fonte; a coluna "Evidência" é o trecho que sustenta.

| Chave | DOI | Afirmação usada no artigo | Evidência na fonte |
|---|---|---|---|
| clement2010 | 10.1109/TPWRS.2009.2036481 | Carregamento não coordenado causa problemas na rede de distribuição (perdas, desvios de tensão); coordenação é proposta | Resumo: "This uncoordinated power consumption on a local scale can lead to grid problems. Therefore, coordinated charging is proposed..." |
| garcia2014 | 10.1016/j.rser.2014.07.040 | Revisão de estratégias/algoritmos de smart charging; gestão inteligente limita custos de reforço da rede, desvios de tensão e saturação de transformadores | Resumo parcial (repositório EHU/RePEc): "implementation of an intelligent management system will be necessary to limit network reinforcement costs and problems such as voltage deviations and transformer saturation" |
| hilshey2013 | 10.1109/PESMG.2013.6672925 | Smart charging baseado em temperatura reduz envelhecimento de transformador; esquemas simples (adiar para meia-noite) podem aumentar o envelhecimento | Resumo (OpenAlex) |
| lee2019acn | 10.1145/3307772.3328313 | Dataset aberto de recarga em local de trabalho (>30.000 sessões); dados mostram que a entrada do usuário (saída estimada, energia) é pouco confiável | Resumo + texto completo: "Our data, however, shows that user input can be quite unreliable"; "user input data conspicuously gives the worst overall prediction" |
| lee2021acn | 10.1109/TSG.2021.3074437 | ACN implantada desde 2016 em Caltech e >100 locais; algoritmo baseado em otimização convexa e MPC | Resumo |
| lee2021acnsim | 10.1109/TSG.2021.3103156 | Simulador aberto, orientado a dados, integra ACN-Data e OpenAI Gym para RL | Resumo |
| orfanoudakis2025 | 10.1109/TITS.2024.3510945 | Poucos simuladores de smart charging oferecem ambiente Gym para RL; EV2Gym com modelos validados em dados reais | Resumo: "only a few support the development of Reinforcement Learning (RL) algorithms in the form of a Gym environment" |
| wan2019 | 10.1109/TSG.2018.2879572 | Agendamento de recarga formulado como MDP com probabilidades de transição desconhecidas; DRL model-free com sinais de preço em tempo real | Resumo |
| yan2021 | 10.1109/TSG.2021.3098298 | DRL (SAC) para recarga de um VE individual, com comportamento dinâmico do usuário e preço | Resumo: "individual EV charging scheduling problem" |
| abdullah2021 | 10.1109/ACCESS.2021.3064354 | Revisão de RL para gestão de recarga; incertezas de parâmetros dos VEs, preços e cargas; interesse crescente em abordagens orientadas a dados | Resumo |
| kirk2023 | 10.1613/jair.1.14174 | Generalização zero-shot em RL: políticas devem generalizar a situações novas, evitando overfitting ao ambiente de treino | Resumo |
| sorensen2021data | 10.1016/j.dib.2021.107105 | 6.878 sessões, 97 usuários, dez/2018–jan/2020, prédios residenciais na Noruega | Resumo |
| sorensen2021flex | 10.1016/j.enbuild.2021.110923 | Potencial de flexibilidade da recarga residencial em prédios de apartamentos | Resumo |
| henderson2018 | 10.1609/aaai.v32i1.11694 | Reprodutibilidade em DRL: não determinismo e variância dificultam comparações; recomendam métricas de significância | Resumo |
| akiba2019 | 10.1145/3292500.3330701 | Optuna: busca e poda de hiperparâmetros | Resumo |
| tobin2017 | 10.1109/IROS.2017.8202133 | Domain randomization: com variabilidade suficiente no simulador, o mundo real aparece como mais uma variação | Resumo |
| li2023ood (arXiv, sem DOI) | arXiv:2311.05941 | Algoritmos de aprendizado treinados com dados pré-COVID degradam no pós-COVID (mudança de distribuição) | Página do arXiv / resumo; autores Tongxin Li, Chenxi Sun |
| zaheer2017, vaswani2017, schulman2017, ng1999 | sem DOI (NeurIPS/arXiv/ICML) | Métodos (DeepSets, atenção, PPO, shaping) | Referências de método |

Removida: arXiv:2410.16425 (survey de RL para recarga) — **retirado pelos autores no arXiv**.

Observação: o Consensus atingiu o limite mensal (30/30, renova em 1/11/2026); a busca foi feita
por busca web + Crossref + Semantic Scholar/OpenAlex.

## Usos acrescentados na versão de submissão (27/10/2026)

DOIs reconferidos no Crossref em 27/10/2026: os 17 DOIs batem em título, primeiro autor, ano, volume e páginas. O arXiv:2311.05941 foi conferido na API do arXiv (v3, autores Tongxin Li e Chenxi Sun).

| Chave | Onde | Frase no artigo | Evidência na fonte |
|---|---|---|---|
| li2023ood | Introdução e Seção V-B | Li e Sun descrevem o MPC como conservador e independente dos dados e o RL como confiando totalmente nos dados; avaliam um escalonador OOD-aware sob a mudança de comportamento causada pela COVID-19 no ACN-Data | Resumo: "MPC is often too conservative and data-independent, whereas RL tends to be overly aggressive and fully trusts the data"; "remarkable shifts of EV charging behaviors caused by COVID-19 in the Caltech ACN-Data" |
| lee2021acn | Introdução, Protocolo, Conclusão | Infraestrutura sobre-subscrita; MPC baseado em otimização convexa; desafios práticos: trifásico desbalanceado, bateria não ideal, sinais de controle quantizados | Resumo: "allows for significant over-subscription of electrical infrastructure"; "unbalanced three-phase infrastructure, non-ideal battery charging behavior, and quantized control signals" |
| wan2019modelfree | Introdução | Recarga sob preços variáveis no tempo | Resumo: DRL com sinais de preço em tempo real |
| sorensen2021flex | Seção V-A | Potencial de flexibilidade em prédios de apartamentos com pontos de recarga privados | Resumo: "significant potential for residential EV charging flexibility when private parking spaces have EV charge points" |
| henderson2018matters | Seção V-C | Dependência dos resultados de DRL em relação à semente | Resumo: "variance intrinsic to the methods, can make reported results tough to interpret" |
| kirk2023survey | Seção V-C | Gap de transferência como diferença entre desempenho no treino e no teste | Formalismo de generalização zero-shot do survey (gap de generalização = desempenho em treino − desempenho em teste) |
