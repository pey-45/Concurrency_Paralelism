-module(comp).

%% Funciones visibles por otros archivos, con su cantidad de argumentos
-export([comp_proc/2, comp_proc/3 , decomp_proc/2, decomp_proc/3]).

-define(DEFAULT_CHUNK_SIZE, 1024*1024).

%% File Compression
comp_proc(File, Procs) ->
    comp_proc(File, ?DEFAULT_CHUNK_SIZE, Procs).

comp_proc(File, Chunk_Size, Procs) ->  %% Inicializa un Reader y un Writer que corren en procesos separados
    case file_service:start_file_reader(File, Chunk_Size) of
        {ok, Reader} -> %% Se inicia el proceso de lectura
            case archive:start_archive_writer(File++".ch") of
                {ok, Writer} -> %% Se inicia el proceso de escritura
                    comp_loop(Reader, Writer),
                    case Procs of
                        0 -> ok;
                        _ -> comp_proc(File, Chunk_Size, Procs-1)
                    end;
                {error, Reason} ->
                    io:format("Could not open output file: ~w~n", [Reason])
            end;
        {error, Reason} ->
            io:format("Could not open input file: ~w~n", [Reason])
    end.

comp_loop(Reader, Writer) ->  %% Bucle de compresion => coger un chunk, comprimirlo, y enviarselo al Writer
    Reader ! {get_chunk, self()},  %% El proceso de lectura solicita un chunk
    receive
        {chunk, Num, Offset, Data} ->   %% Recibe un chunk, lo comprime, y se lo envia al Writer
            Comp_Data = compress:compress(Data), %% Se almacena el archivo comprimido en Comp_Data
            Writer ! {add_chunk, Num, Offset, Comp_Data}, %% El proceso de escritura recibe el chunk comprimido
            comp_loop(Reader, Writer); %% Recursion
        eof ->  %% Fin del archivo, se detienen el Reader y el Writer
            Writer ! stop;
        {error, Reason} ->
            io:format("Error reading input file: ~w~n",[Reason]),
            Reader ! stop,
            Writer ! abort
    end.

%% File Decompression
decomp_proc(Archive, Procs) ->
    decomp_proc(Archive, string:replace(Archive, ".ch", "", trailing), Procs).

decomp_proc(Archive, Output_File, Procs) ->
    case archive:start_archive_reader(Archive) of
        {ok, Reader} ->
            case file_service:start_file_writer(Output_File) of
                {ok, Writer} ->
                    decomp_loop(Reader, Writer),
                    case Procs of
                        0 -> ok;
                        _ -> decomp_proc(Archive, Output_File, Procs - 1)
                    end;
                {error, Reason} ->
                    io:format("Could not open output file: ~w~n", [Reason])
            end;
        {error, Reason} ->
            io:format("Could not open input file: ~w~n", [Reason])
    end.

decomp_loop(Reader, Writer) ->
    Reader ! {get_chunk, self()},  %% request a chunk from the reader
    receive
        {chunk, _Num, Offset, Comp_Data} ->  %% got one
            Data = compress:decompress(Comp_Data),
            Writer ! {write_chunk, Offset, Data},
            decomp_loop(Reader, Writer);
        eof ->    %% end of file => exit decompression
            Reader ! stop,
            Writer ! stop;
        {error, Reason} ->
            io:format("Error reading input file: ~w~n", [Reason]),
            Writer ! abort,
            Reader ! stop
    end.

