% Pregunta 1 - Archivo pregunta1.m

% Definir los valores de k

function nombrpregunta1()
    clc; clear;
    k_values = [5, 6, 7, 8, 9, 10, 11, 12];
    num_elements = length(k_values);

    tiempo1=[];
    tiempo2=[];

    k = 1:1:num_elements;
    for m = k
        % Generar matriz aleatoria A de tamaño 2^k x 2^(k-1)
        A = rand(2^k_values(m), 2^(k_values(m)-1));

        tic; [U ,S , V ]= svd(A); t1=toc;
        tiempo1=[tiempo1 t1];

         tic; [Ur, Sr, Vr] = svdCompact(A); t2=toc;
         tiempo2=[tiempo2 t2];

    end
    hold on
    plot(k_values(k),tiempo1,'r','LineWidth',2)
    plot(k_values(k),tiempo2,'b','LineWidth',2)
    xlabel('Parámetro K'); ylabel('Tiempo (en segundos)');
    legend('GNU Octave','Nuevo Metodo')
end

function [Ur, Sr, Vr] = svdCompact (A)

    [m,n] = size(A);
    if m > n
        M1 = A' * A;
        [V1, D] = eig(M1);
        y1 = diag(D);
        const = n * max(y1) * eps;
        y2 = (y1 > const);
        rA = sum(y2);
        y3 = y1.*y2; %rango de la matriz
        [s1, order] = sort(sqrt(y3), 'descend');
        V2 = V1(:, order);
        Vr = V2(:, 1: rA);
        Sr = diag(s1(1: rA));
        Ur = (1./(s1(1:rA))').*(A * Vr);
    else
        M1 = A * A';
        [U1, D] = eig(M1);
        y1 = diag(D);
        const = m * max(y1) * eps;
        y2 = (y1 > const);
        rA = sum(y2); %rango de la matriz
        y3 = y1.*y2;
        [s1, order] = sort(sqrt(y3), 'descend');
        U2 = U1(:, order);
        Ur = U2(:, 1: rA);
        Sr = diag(s1(1 : rA));
        Vr = (1./(s1(1 : rA))').*(A'*Ur);
    end

endfunction






