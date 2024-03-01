%%Cargar datos
file='./Datos/datos_tip_year.csv';
data_year=readtable(file,'Delimiter',',', 'DecimalSeparator','.', 'VariableNamingRule','preserve');
  Tnoc=51.4;
%escoger n_it filas diferentes
n_mod=5;
n_it=100;
if n_it>=size(data_year,1)
    n_it=size(data_year,1);
end
rows_data=unique(round((size(data_year,1)-1).*rand(n_it,1) + 1));

while n_it>size(rows_data,1)
    rows_data(end+1)=round((size(data_year,1)-1).*rand + 1);
    rows_data=unique(rows_data);
end
V_mod=[];
I_mod=[];
Ir_mod=[];
Tstart=tic;
for i=1:n_it
    fprintf('Iteracion %d de %d \n',i,n_it);
    Ir_total=data_year.Ir_total(rows_data(i));
    Ir_difusa=data_year.Ir_difusa(rows_data(i));
    Tamb=data_year.Temperatura(rows_data(i));
    Ir(1)=Ir_total;
    T(1)=Tamb+Ir(1)*((Tnoc-20)/800);
    for j=2:n_mod
        Ir(j)=Ir_difusa+(Ir_total-Ir_difusa)*rand;
        T(j)=Tamb+Ir(j)*((Tnoc-20)/800);
    end
    out=sim('Obtencion_Datos');
    if i==1
        V_mod=[out.V1,out.V2,out.V3,out.V4,out.V5];
        I_mod=[out.I1,out.I2,out.I3,out.I4,out.I5];
        for k=1:size(out.V1,1)
        Ir_mod(k,:)=Ir;
        T_mod(k,:)=T;
        end 
    else
         aux_V_mod=[out.V1,out.V2,out.V3,out.V4,out.V5];
         V_mod=[V_mod; aux_V_mod];
         aux_I_mod=[out.I1,out.I2,out.I3,out.I4,out.I5];
          I_mod=[I_mod; aux_I_mod];
          for k=1:size(out.V1,1)
              aux_Ir_mod(k,:)=Ir;
              aux_T_mod(k,:)=T;
          end
          
         Ir_mod=[Ir_mod; aux_Ir_mod];
         T_mod=[T_mod; aux_T_mod];
    end 
    
    
%     V1(i,:)=ans.V1;
%     V2(i,:)=out.V2;
%     V3(i,:)=ansV3;
%     V4(i,:)=out.V4;
%     V5(i,:)=out.V5;
%     Vtotal(i,:)=out.VTotal;
%     Itotal(i,:)=out.Itotal;
%      I1(i,:)=out.I1;
%     I2(i,:)=out.I2;
%     I3(i,:)=out.I3;
%     I4(i,:)=out.I4;
%     I5(i,:)=out.I5;
   clear aux_I_mod aux_Ir_mod aux_V_mod Ir T out
end
tend=toc(Tstart);
fprintf('tiempo= %f',tend);
writetable(array2table(V_mod),'./Datos/V_mod.csv');
writetable(array2table(I_mod),'./Datos/I_mod.csv');
writetable(array2table(Ir_mod),'./Datos/Ir_mod.csv');
writetable(array2table(T_mod),'./Datos/T_mod.csv');

%T=Tamb+G*(Tnoc-20)/800;



%%Iniciar iteraciones

%%Asignar datos a variables

%%Iniciar la simulación

%%Cargar los datos al final de cada iteración

%%Finalizar