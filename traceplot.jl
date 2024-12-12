using Plots

# Funzione per creare grafici
function traceplot(chain, chain_name, burn_in)

    if ndims(chain[1]) == 0
        
        n = length(chain)

        p = plot()

        plot!(p, 1:n, chain, title="traceplot di $(chain_name)")

        vline!(p, [burn_in], label="Burn-in", linestyle=:dash, color=:red)

        display(p)

    else

        n = length(chain)
        k = length(chain[1])

        # Numero massimo di subplot per immagine
        max_plots_per_image = 4

        # Calcolo del numero di immagini necessarie
        n_images = ceil(Int, k / max_plots_per_image)

        for img_idx in 1:n_images
            # Determina gli indici degli elementi per questa immagine
            start_idx = (img_idx - 1) * max_plots_per_image + 1
            end_idx = min(img_idx * max_plots_per_image, k)
            indices = start_idx:end_idx

            # Layout dei subplot per questa immagine
            layout = @layout [grid(2, 2)]  # Grafici in colonna

            # Crea il plot
            p = plot(layout=layout, title="traceplot di $(chain_name), Immagine $img_idx")

            for (i, idx) in enumerate(indices)
                # Aggiungi il variogramma per l'elemento corrente
                plot!(p, 1:n, [chain[temp][idx] for temp in 1:n], 
                    title="traceplot dell'elemento $idx di $(chain_name)", 
                    subplot=i)

                # Aggiungi la linea del burn-in
                vline!(p, [burn_in], label="Burn-in", linestyle=:dash, color=:red, subplot=i)
            end

            display(p)

        end


    end
end

function optimal_grid(n)
    # Calcola il numero di righe e colonne
    rows = floor(Int, sqrt(n))
    cols = ceil(Int, sqrt(n))

    # Se il prodotto non copre tutti i grafici, aumenta il numero di righe
    if rows * cols < n
        rows += 1
    end

    return rows, cols
end
