"""Controlador dos Fluxos 1/2/3 — orquestração de alto nível do ciclo.

É aqui que mora a sequência completa: gerar o design (doe/), expandir em
plano de execução, escrever o cockpit .xlsx (sim_interface/), e — depois
que o ProModel devolve a planilha preenchida — ler, agregar e seguir para
metamodelagem. Não existe execução em lote do lado do Python: quem
percorre as linhas rodando a simulação é o ProModel, a partir do cockpit.
"""
