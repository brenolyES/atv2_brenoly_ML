{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/brenolyES/atv2_brenoly_ML/blob/main/Brenoly_Atividade_com_chatGPT.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Importação dos dados"
      ],
      "metadata": {
        "id": "o-XrbXguoXkM"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iUxwpHrdn-fp"
      },
      "outputs": [],
      "source": [
        "#importar as bibliotecas necessárias\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "url = \"https://gist.githubusercontent.com/brenolyES/6f34e3f24dfe4992ca71c8bdeee4fa3f/raw/a31b3d2acd6db4ae8bcc954f3d7a275f3b35f46a/gistfile1.txt\"\n",
        "df = pd.read_csv(url, sep=';')\n"
      ],
      "metadata": {
        "id": "1sMJoA1go4hY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Importando a base e separando as colunas pela virgula."
      ],
      "metadata": {
        "id": "pl7Xf1WPZx3_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#analisando as primeiras entradas do df\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "C0XtAtPQo_uR",
        "outputId": "97b69c2e-1835-4d61-b555-bde1034cd815"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   player_id               name  nationality  position  overall  age  hits  \\\n",
              "0     158023       Lionel Messi    Argentina  ST|CF|RW       94   33   299   \n",
              "1      20801  Cristiano Ronaldo     Portugal     ST|LW       93   35   276   \n",
              "2     190871          Neymar Jr       Brazil    CAM|LW       92   28   186   \n",
              "3     203376    Virgil van Dijk  Netherlands        CB       91   29   127   \n",
              "4     200389          Jan Oblak     Slovenia        GK       91   27    47   \n",
              "\n",
              "   potential                  team  \n",
              "0         94         FC Barcelona   \n",
              "1         93             Juventus   \n",
              "2         92  Paris Saint-Germain   \n",
              "3         92            Liverpool   \n",
              "4         93      Atlético Madrid   "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2663609f-508f-488b-afee-04562f2021f6\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>player_id</th>\n",
              "      <th>name</th>\n",
              "      <th>nationality</th>\n",
              "      <th>position</th>\n",
              "      <th>overall</th>\n",
              "      <th>age</th>\n",
              "      <th>hits</th>\n",
              "      <th>potential</th>\n",
              "      <th>team</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>158023</td>\n",
              "      <td>Lionel Messi</td>\n",
              "      <td>Argentina</td>\n",
              "      <td>ST|CF|RW</td>\n",
              "      <td>94</td>\n",
              "      <td>33</td>\n",
              "      <td>299</td>\n",
              "      <td>94</td>\n",
              "      <td>FC Barcelona</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20801</td>\n",
              "      <td>Cristiano Ronaldo</td>\n",
              "      <td>Portugal</td>\n",
              "      <td>ST|LW</td>\n",
              "      <td>93</td>\n",
              "      <td>35</td>\n",
              "      <td>276</td>\n",
              "      <td>93</td>\n",
              "      <td>Juventus</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>190871</td>\n",
              "      <td>Neymar Jr</td>\n",
              "      <td>Brazil</td>\n",
              "      <td>CAM|LW</td>\n",
              "      <td>92</td>\n",
              "      <td>28</td>\n",
              "      <td>186</td>\n",
              "      <td>92</td>\n",
              "      <td>Paris Saint-Germain</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>203376</td>\n",
              "      <td>Virgil van Dijk</td>\n",
              "      <td>Netherlands</td>\n",
              "      <td>CB</td>\n",
              "      <td>91</td>\n",
              "      <td>29</td>\n",
              "      <td>127</td>\n",
              "      <td>92</td>\n",
              "      <td>Liverpool</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>200389</td>\n",
              "      <td>Jan Oblak</td>\n",
              "      <td>Slovenia</td>\n",
              "      <td>GK</td>\n",
              "      <td>91</td>\n",
              "      <td>27</td>\n",
              "      <td>47</td>\n",
              "      <td>93</td>\n",
              "      <td>Atlético Madrid</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2663609f-508f-488b-afee-04562f2021f6')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2663609f-508f-488b-afee-04562f2021f6 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2663609f-508f-488b-afee-04562f2021f6');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "aXwCPOKFMOKT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df[\"position\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D2DTIZL0a1Z6",
        "outputId": "76a205f0-fe71-4960-f456-943e52ffeaf3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "CB               2296\n",
              "GK               1884\n",
              "ST               1757\n",
              "CDM|CM           1546\n",
              "LB                695\n",
              "                 ... \n",
              "LB|RW               1\n",
              "LWB|CM|CAM|LW       1\n",
              "RB|RW|LW            1\n",
              "CB|CF               1\n",
              "CB|RM|LM            1\n",
              "Name: position, Length: 232, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pode-se obersevar que um jogador pode ter mais do que uma posição dentro do jogo e da sua função no time."
      ],
      "metadata": {
        "id": "t2u2SPKQbImF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby('position').mean()"
      ],
      "metadata": {
        "id": "egtgLsa9MPWS",
        "outputId": "3b4c5df2-50fa-4a76-ed73-a09ac8988986",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 455
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  player_id    overall        age      hits  potential\n",
              "position                                                              \n",
              "CAM           231387.455357  65.602679  24.575893  0.959821  71.758929\n",
              "CAM|CF        230731.895833  67.645833  24.979167  5.416667  73.604167\n",
              "CAM|CF|LW     218185.500000  69.700000  26.300000  9.700000  75.100000\n",
              "CAM|CF|RW     224962.833333  68.500000  24.500000  2.166667  76.166667\n",
              "CAM|LW        217544.500000  70.477273  26.045455  8.159091  74.772727\n",
              "...                     ...        ...        ...       ...        ...\n",
              "ST|RM|RW      220465.794872  68.230769  25.948718  2.897436  72.256410\n",
              "ST|RW         226349.421488  66.561983  24.966942  2.958678  72.520661\n",
              "ST|RWB|RM|RW  236393.666667  64.000000  28.000000  0.000000  66.333333\n",
              "ST|RWB|RW     198982.000000  59.000000  26.500000  0.500000  62.000000\n",
              "ST|RW|LW      227402.549020  67.813725  25.078431  5.774510  73.441176\n",
              "\n",
              "[232 rows x 5 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-37d2bb34-3fef-4110-8141-4fce2d1cdb0a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>player_id</th>\n",
              "      <th>overall</th>\n",
              "      <th>age</th>\n",
              "      <th>hits</th>\n",
              "      <th>potential</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>position</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>CAM</th>\n",
              "      <td>231387.455357</td>\n",
              "      <td>65.602679</td>\n",
              "      <td>24.575893</td>\n",
              "      <td>0.959821</td>\n",
              "      <td>71.758929</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>CAM|CF</th>\n",
              "      <td>230731.895833</td>\n",
              "      <td>67.645833</td>\n",
              "      <td>24.979167</td>\n",
              "      <td>5.416667</td>\n",
              "      <td>73.604167</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>CAM|CF|LW</th>\n",
              "      <td>218185.500000</td>\n",
              "      <td>69.700000</td>\n",
              "      <td>26.300000</td>\n",
              "      <td>9.700000</td>\n",
              "      <td>75.100000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>CAM|CF|RW</th>\n",
              "      <td>224962.833333</td>\n",
              "      <td>68.500000</td>\n",
              "      <td>24.500000</td>\n",
              "      <td>2.166667</td>\n",
              "      <td>76.166667</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>CAM|LW</th>\n",
              "      <td>217544.500000</td>\n",
              "      <td>70.477273</td>\n",
              "      <td>26.045455</td>\n",
              "      <td>8.159091</td>\n",
              "      <td>74.772727</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST|RM|RW</th>\n",
              "      <td>220465.794872</td>\n",
              "      <td>68.230769</td>\n",
              "      <td>25.948718</td>\n",
              "      <td>2.897436</td>\n",
              "      <td>72.256410</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST|RW</th>\n",
              "      <td>226349.421488</td>\n",
              "      <td>66.561983</td>\n",
              "      <td>24.966942</td>\n",
              "      <td>2.958678</td>\n",
              "      <td>72.520661</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST|RWB|RM|RW</th>\n",
              "      <td>236393.666667</td>\n",
              "      <td>64.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>66.333333</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST|RWB|RW</th>\n",
              "      <td>198982.000000</td>\n",
              "      <td>59.000000</td>\n",
              "      <td>26.500000</td>\n",
              "      <td>0.500000</td>\n",
              "      <td>62.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST|RW|LW</th>\n",
              "      <td>227402.549020</td>\n",
              "      <td>67.813725</td>\n",
              "      <td>25.078431</td>\n",
              "      <td>5.774510</td>\n",
              "      <td>73.441176</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>232 rows × 5 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-37d2bb34-3fef-4110-8141-4fce2d1cdb0a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-37d2bb34-3fef-4110-8141-4fce2d1cdb0a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-37d2bb34-3fef-4110-8141-4fce2d1cdb0a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "A tabela resultante mostra a média de cada variável numérica para cada combinação de posições listadas na coluna \"position\". Por exemplo, para jogadores que têm a posição listada como \"CAM\", a média de overall é 65.6, a média de idade é 24.6, a média de hits é 0.96, e a média de potential é 71.76. As categorias de posição foram criadas com base nas informações da coluna \"position\", que contém várias posições separadas por \"|\". Cada combinação única de posições se tornou uma categoria separada."
      ],
      "metadata": {
        "id": "dKMzkTq0aFGm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9qatpM3vpOMW",
        "outputId": "2cac3276-d967-4bb9-d52a-33a152e0137d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 17981 entries, 0 to 17980\n",
            "Data columns (total 9 columns):\n",
            " #   Column       Non-Null Count  Dtype \n",
            "---  ------       --------------  ----- \n",
            " 0   player_id    17981 non-null  int64 \n",
            " 1   name         17981 non-null  object\n",
            " 2   nationality  17981 non-null  object\n",
            " 3   position     17981 non-null  object\n",
            " 4   overall      17981 non-null  int64 \n",
            " 5   age          17981 non-null  int64 \n",
            " 6   hits         17981 non-null  int64 \n",
            " 7   potential    17981 non-null  int64 \n",
            " 8   team         17981 non-null  object\n",
            "dtypes: int64(5), object(4)\n",
            "memory usage: 1.2+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Podemos observar que essa base nao possue nenhum valor nulo entre os dados;"
      ],
      "metadata": {
        "id": "6e4N0R12aKNa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "vTOAVreQsli3",
        "outputId": "8442f0b1-afb6-4726-b188-6753aba7b8fc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "           player_id       overall           age          hits     potential\n",
              "count   17981.000000  17981.000000  17981.000000  17981.000000  17981.000000\n",
              "mean   220912.660531     67.274345     26.311440      2.689450     71.738057\n",
              "std     27738.072671      5.924392      4.556077     10.846286      5.961968\n",
              "min        41.000000     56.000000     17.000000      0.000000     57.000000\n",
              "25%    204881.000000     63.000000     23.000000      0.000000     67.000000\n",
              "50%    226753.000000     67.000000     26.000000      0.000000     71.000000\n",
              "75%    241587.000000     71.000000     30.000000      2.000000     76.000000\n",
              "max    256469.000000     94.000000     43.000000    371.000000     95.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-489ff4d3-75e5-466a-b5a0-7014d9b8f6a4\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>player_id</th>\n",
              "      <th>overall</th>\n",
              "      <th>age</th>\n",
              "      <th>hits</th>\n",
              "      <th>potential</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>17981.000000</td>\n",
              "      <td>17981.000000</td>\n",
              "      <td>17981.000000</td>\n",
              "      <td>17981.000000</td>\n",
              "      <td>17981.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>220912.660531</td>\n",
              "      <td>67.274345</td>\n",
              "      <td>26.311440</td>\n",
              "      <td>2.689450</td>\n",
              "      <td>71.738057</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>27738.072671</td>\n",
              "      <td>5.924392</td>\n",
              "      <td>4.556077</td>\n",
              "      <td>10.846286</td>\n",
              "      <td>5.961968</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>41.000000</td>\n",
              "      <td>56.000000</td>\n",
              "      <td>17.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>57.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>204881.000000</td>\n",
              "      <td>63.000000</td>\n",
              "      <td>23.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>67.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>226753.000000</td>\n",
              "      <td>67.000000</td>\n",
              "      <td>26.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>71.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>241587.000000</td>\n",
              "      <td>71.000000</td>\n",
              "      <td>30.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>76.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>256469.000000</td>\n",
              "      <td>94.000000</td>\n",
              "      <td>43.000000</td>\n",
              "      <td>371.000000</td>\n",
              "      <td>95.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-489ff4d3-75e5-466a-b5a0-7014d9b8f6a4')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-489ff4d3-75e5-466a-b5a0-7014d9b8f6a4 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-489ff4d3-75e5-466a-b5a0-7014d9b8f6a4');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.hist(bins=50, figsize=(20,15))\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 879
        },
        "id": "Lht28t32uo9E",
        "outputId": "8493e7d4-1da3-402f-cdd4-b150b5ef7580"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1440x1080 with 6 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pode mos observar que a maioria dos jogadores tem o overall entre 65 a 70, a maioria dos jogadores estao entre 23 a 25 anos, poucos jogadores tem a taxa de acerto alta e que a maioria dos jogadores estao com potencial entre 65 a 75."
      ],
      "metadata": {
        "id": "SPYtE_wSZGxf"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Usando o CrossValidation no modelo**"
      ],
      "metadata": {
        "id": "VWLAiBXTYwYm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import KFold\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Selecionar os recursos de entrada (X) e a variável alvo (y)\n",
        "X = df[['age', 'hits', 'potential']]\n",
        "y = df['overall']\n",
        "\n",
        "# Definir o número de folds para a validação cruzada k-fold\n",
        "n_folds = 5\n",
        "\n",
        "# Inicializar o objeto de validação cruzada k-fold\n",
        "kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)\n",
        "\n",
        "# Inicializar o modelo de regressão linear\n",
        "model = LinearRegression()\n",
        "\n",
        "# Realizar a validação cruzada k-fold\n",
        "for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):\n",
        "    print(f'Fold {fold+1}:')\n",
        "    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]\n",
        "    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]\n",
        "    model.fit(X_train, y_train)\n",
        "    score = model.score(X_val, y_val)\n",
        "    print(f'R2 score: {score:.2f}')\n"
      ],
      "metadata": {
        "id": "mvCjX-OEPoXO",
        "outputId": "8dc85903-ee99-48dd-f9fe-511fca7efe2b",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fold 1:\n",
            "R2 score: 0.81\n",
            "Fold 2:\n",
            "R2 score: 0.81\n",
            "Fold 3:\n",
            "R2 score: 0.80\n",
            "Fold 4:\n",
            "R2 score: 0.80\n",
            "Fold 5:\n",
            "R2 score: 0.81\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Os resultados mostram o desempenho do modelo em cada uma das cinco dobras (folds) do conjunto de dados durante a validação cruzada. A medida de desempenho usada foi o R2 score, que é uma medida de quão bem o modelo se ajusta aos dados.\n",
        "\n",
        "Os valores de R2 score variam de 0 a 1, onde 0 indica que o modelo não explica nada da variância dos dados, e 1 indica que o modelo explica toda a variância dos dados. Nesse caso, os resultados indicam que o modelo tem um bom desempenho, com valores de R2 score variando entre 0.80 e 0.81 em todas as dobras."
      ],
      "metadata": {
        "id": "rLdki7EOYtPN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Usando o GridSearch no modelo**"
      ],
      "metadata": {
        "id": "H2qbA4TSY6Qt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "# Define the pipeline with StandardScaler() and LinearRegression()\n",
        "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
        "\n",
        "# Define the hyperparameters to be tested\n",
        "param_grid = {'linearregression__fit_intercept': [True, False]}\n",
        "\n",
        "# Create a GridSearch object with the model and the hyperparameters\n",
        "grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)\n",
        "\n",
        "# Train the model with GridSearch\n",
        "grid_search.fit(X, y)\n",
        "\n",
        "# Get the best set of hyperparameters and the best score\n",
        "print(grid_search.best_params_)\n",
        "print(grid_search.best_score_)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "M8OpWgKbV2wr",
        "outputId": "03fd7c59-10eb-4f95-cdde-9bfb085dda92"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'linearregression__fit_intercept': False}\n",
            "-3.8506772143326997\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Esse resultado indica que, dentre os hiperparâmetros testados, o melhor conjunto de hiperparâmetros para o modelo de regressão linear é aquele em que o hiperparâmetro fit_intercept é definido como False. Além disso, o valor -3.8506772143326997 representa o melhor score obtido pelo modelo com esse conjunto de hiperparâmetros, durante a busca do GridSearch. Esses resultados indicam que, de acordo com a métrica usada (provavelmente R², já que o modelo é de regressão linear), o modelo obteve um desempenho melhor quando o hiperparâmetro fit_intercept foi definido como False."
      ],
      "metadata": {
        "id": "4nD47suQYeFe"
      }
    }
  ]
}