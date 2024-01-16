%Paulo Régis P. Lima%
clear,clc
for T = 1:5 % Serão realizados 5 treinamentos

    % Obter o conjunto de amostras de treinamento
    dados = load('dados_de_treinamento.dat');
    quantidade_dados_de_treinamento = size(dados,1);
    dados = [linspace(-1,-1,quantidade_dados_de_treinamento)' dados];
    x = dados(:,1:4);

    % Associar a saída desejada para cada amostra obtida
    d = dados(:,5);

    % Inicializar as matrizes pesos W1 (Neuronios Intermediário x Entradas) e
    % W2(Neurônios saida x Neuronios Intermediários+1) aleatoriamente com
    % valores aleatórios pequenos
    neuronios_intermediarios = 10;
    neuronios_saida = 1;
    entradas = size(x,2);

    w1 = random('Uniform',0,0.2,neuronios_intermediarios,entradas);
    w2 = random('Uniform',0,0.2,neuronios_saida,neuronios_intermediarios+1);
    titulo = 'Sem momentum: ';
    % Salvar as matrizes iniciais w1 e w2
    save(strcat('T',num2str(T),' - matrizes iniciais'),'w1','w2')

    % Taxa de aprendizagem (ta) e precisão
    ta        = 0.1;
    precisao  = 10^-6;

    % Iniciar o contador de épocas
    ep = 1;

    % Iniciar o Erro Quadrático Médio atual
    EQM = 0;

    % Laço principal
    while true
        for k = 1:length(x)
         % Fase Foward
         % 10x1  10x4  4x1     10x1              10x1  11x1
            I1  = w1 * x(k,:)'; Y1 = 1./(1 + exp(-I1)); Y1 = [-1; Y1];
         % 1x1   1x11 11x1    1x1               1x1
            I2  = w2 * Y1;    Y2 = 1./(1 + exp(-I2));

         % Fase backward
         %Derivada da função sigmóide em I1
         % 10x1     10x1         10x1
            a = exp(-I1)./(1+exp(-I1)).^2;
         %Derivada da função sigmóide em I2
         % 1x1      1x1          1x1
            b = exp(-I2)./(1+exp(-I2)).^2;
         %    1x1     1x1    1x1    1x1
            delta2   = b .* (d(k)'-Y2);
         %  1x11 1x11   1x1    1x1     1x11
            w2 =  w2 + (ta * delta2) * Y1';
         %   10x1     10x1   [1x1    1x10]'
            delta1   = a .* (delta2'*w2(:,2:neuronios_intermediarios+1))';
         % 10x4  10x4  1x1    10x1    1x4
            w1 =  w1 + ta *  delta1*x(k,:);
        end
% Obter saída da rede ajustada
        for k = 1:length(x)
         % 10x1 10x4   4x1    10x1              10x1  11x1
            I1 = w1 * x(k,:)'; Y1 = 1./(1 + exp(-I1)); Y1 = [-1; Y1];
         % 1x1  1x11  1x1    1x1               1x1
            I2 = w2 * Y1;    Y2 = 1./(1 + exp(-I2));
         %   1x1             1x1  1x1
            EQ(:,k) = 0.5*((d(k)'-Y2).^2);
        end

        % Cálculo do EQM
      % 1x1   1x1       1x1   1x1
        EQM = EQM + sum(EQ)/length(x);
        ep = ep + 1;
      % 1x1         1x1
        eqm(:,ep) = EQM;
        EQM = 0;
      %          1x1          1x1        1x1
        if (abs(eqm(ep) - eqm(ep-1)) < precisao)
            break
        end

    end
    disp('Treinamento finalizado! ')

    %% Validação
    % Obter o conjunto de dados de validação
    validacao = load('dados_de_validacao.dat');
    quantidade_dados_de_validacao = size(validacao,1);
    validacao = [linspace(-1,-1,quantidade_dados_de_validacao)' validacao];

    xv = validacao(:,1:4);
    yv = validacao(:,5);

    % Fase Foward
    % 10x20 10x4 4x20 10x20           10x20   11x20          1x20        10x20
       I1  = w1 * xv'; Y1 = 1./(1 + exp(-I1)); Y1 = [-ones(1,size(xv,1)); Y1];
    % 1x20  1x11 11x20  1x20              1x20
       I2  = w2 * Y1;    Y2 = 1./(1 + exp(-I2));

    % Gráficos do EQM
    plot(eqm(:,2:size(eqm,2))),grid
    grafico = gca;
    xlabel('Épocas'),ylabel('EQM'),title(strcat('Treinamento T',num2str(T)))
    % Salvar o gráfico do Épocas x EQM
   saveas(grafico, strcat('T',num2str(T),' - EQM','.jpg'), 'jpg')

    % Dados de saída
    EQM_final = eqm(ep)
    Epocas = ep
    Saida_da_rede = Y2'
    % Erro Relativo Médio percentual ((Calculado-Real)/Real)/quantidade
    ER = 100*((yv - Y2')./yv);
    ERM = sum(ER)/length(yv)
    % Variância do Erro Relativo (Somatório[(Xi-Xméd)^2])/N
    Variancia = var(ER)
    % Salvar os dados finais
    save(strcat('T',num2str(T),' - Dados de saída'),'EQM_final','Epocas','Saida_da_rede','ERM','Variancia')
    clear,clc,close
end

