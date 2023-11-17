%
% Funcion de reconocimiento facial donde se utiliza un directorio de entrenamiento donde contiene los directorios de cada persona
% y otro directorio donde se encuentran las imagenes que se desean comparar.
%
function nombrpregunta3 ()
  clc; clear;

%  Datos generales
  sizeLenght = 40;
  numImages = 9;
  totalImg = 360;
  totalPixels = 10304;


  % Calculo de la matriz S con cada una de las imagenes cara
  S = zeros(totalPixels, totalImg);
  columna_s = 1;
  for k = 1:sizeLenght
    for m = 1:numImages
      directory=['training/s',num2str(k),'/',num2str(m),'.jpg'];
      A=imread(directory);
      B=im2double(A);
      x = B(:);
      S(:, columna_s) = x;
      columna_s = columna_s + 1;
    end
  end

  % Calculo de f promedio sumando las columnas
  fProm = zeros(totalPixels, 1);
  for m = 1:totalPixels
      fProm(m) = sum(S(m, :));
  endfor
  fProm = (1/totalImg) * fProm;

  % Calculo de la matriz A restando la columna con fProm
  A = zeros(totalPixels, totalImg);
  for m = 1:totalImg
      A(:,m) = S(:,m) - fProm;
  endfor

  % Descomposicion de la matriz A por medio de svdCompact
  [U, S, V] = svdCompact(A);

  % Comparacion del X de la imagen nueva con el X_i de las imagenes buscada
  for i = 1: sizeLenght;

    correctImage = '';
    minValue = 0;
    compareDirectory = ['compare/p',num2str(i),'.jpg'];
    A_compare = imread(compareDirectory);
    B_compare =im2double(A_compare);
    f = B_compare(:);
    X = U'*(f - fProm);

    for k = 1:sizeLenght
      for m = 1:numImages
        trainingDirectory = ['training/s',num2str(k),'/',num2str(m),'.jpg'];
        A_training = imread(trainingDirectory);
        B_training=im2double(A_training);
        f_i = B_training(:);
        X_i = U'*(f_i - fProm);
        epsilon = ((X - X_i)'*(X - X_i))^(1/2);
        if minValue == 0;
          minValue = epsilon;
        elseif epsilon < minValue;
          minValue = epsilon;
          correctImage = trainingDirectory;
        end
      end
    end

    % Graficacion de las imagenes
    subplot(1,2,1);
    imshow(A_compare)
    title('Rostro nuevo')

    readImage = imread(correctImage);
    subplot(1,2,2);
    imshow(readImage)
    title('Rostro identificado')
    pause(1)

  endfor

endfunction







