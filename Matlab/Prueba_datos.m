%Modificacion del csv de solar
direccion=uigetdir(matlabroot);
listing=dir(direccion);
filenames={listing.name};
for i=1:1:2
filenames(1)=[];
end

for i=1:size(filenames,1)
    k=1;
    filename=fullfile(direccion,filenames(i));
    datos_og=readtable([filename{1,1}]);
    
    %eliminar filas sin datos de irradiancia
    for i=1:size(datos_og,1)
        if datos_og.G_h_(i)==0.0%encontrar las posiciones con 0
            index(k)=i;
            k=k+1;
        end
    end
    
    for  i=1:size(index,2)% eliminarlas
        if i==1
            datos_og(index(1,i),:)=[];
            k=k+1;
        else
            datos_og(index(1,i)-i+1,:)=[];
            k=k+1;
        end
    end
    for i=(size(datos_og,1)-9):(size(datos_og,1)) %eliminar el final de documento con texto
        datos_og(size(datos_og,1),:)=[];
    end
   %convertir casilla time en en mes dia y hora
   
   %tabla de destino
    
    for i=1:size(datos_og,1)
        aux=char(datos_og.time_UTC_(i));
        mes(i,1)=str2num(string(aux(5:6)));
         dia(i,1)=str2num(string(aux(7:8)));
          hora(i,1)=str2num(string(aux(10:11)));
          t=table(mes,dia,hora);
%         T=renamevars(T,'Var1)
    end
    t(:,4)=datos_og(:,4);
     t(:,5)=datos_og(:,6);
     t(:,6)=datos_og(:,2);
    t=renamevars(t,'Var4','Ir_total');
     t=renamevars(t,'Var5','Ir_difusa');
     t=renamevars(t,'Var6','Temperatura');
     direccion=fullfile(direccion,'datos_tip_year.csv');
     writetable(t,direccion)
end


%Obtención de datos de Irradiancia total

%Obtencion de datos de temperatura de aire
