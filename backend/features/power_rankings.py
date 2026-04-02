"""Dynamic league Power Rankings from mean team Elo per league per day.

Normalize all clubs 0-100 globally daily.
Store per-league mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th).
Compute relative_ability = team_score - league_mean_score.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from backend.data import cache, clubelo_client, elo_router, worldfootballelo_client
from backend.utils.league_registry import LEAGUES, LeagueInfo


_ONE_DAY = 86400


# ── ClubElo → Sofascore direct name mapping ──────────────────────────────────
# ClubElo uses abbreviated team names (e.g. "PSG", "ManCity") while Sofascore
# uses full display names (e.g. "Paris Saint-Germain", "Manchester City").
# This mapping canonicalizes ClubElo names at data-load time so that the
# ``teams`` dict keys match the names that come from the Sofascore dropdowns.
# Fuzzy matching is kept as a safety net for any unmapped teams.
_CLUBELO_TO_SOFASCORE: Dict[str, str] = {
    # England
    "ManCity": "Manchester City",
    "ManUtd": "Manchester United",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "NottmForest": "Nottingham Forest",
    "SheffUtd": "Sheffield United",
    "WestHam": "West Ham United",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Leicester": "Leicester City",
    "WestBrom": "West Bromwich Albion",
    "SheffWed": "Sheffield Wednesday",
    "QPR": "Queens Park Rangers",
    "Boro": "Middlesbrough",
    "Coventry": "Coventry City",
    "Stoke": "Stoke City",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Norwich": "Norwich City",
    "Leeds": "Leeds United",
    "Sunderland": "Sunderland AFC",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Bournemouth": "AFC Bournemouth",
    # France
    "PSG": "Paris Saint-Germain",
    "Monaco": "AS Monaco",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Lille": "Lille OSC",
    "Rennes": "Stade Rennais",
    "Nantes": "FC Nantes",
    "Nice": "OGC Nice",
    "Lens": "RC Lens",
    "Strasbourg": "RC Strasbourg Alsace",
    "StEtienne": "AS Saint-Étienne",
    "Montpellier": "Montpellier HSC",
    "Brest": "Stade Brestois 29",
    "Toulouse": "Toulouse FC",
    "Reims": "Stade de Reims",
    "Angers": "Angers SCO",
    "LeHavre": "Le Havre AC",
    # Germany
    "Bayern": "Bayern Munich",
    "BayernMunich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Leipzig": "RB Leipzig",
    "Frankfurt": "Eintracht Frankfurt",
    "Gladbach": "Borussia Mönchengladbach",
    "Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "Stuttgart": "VfB Stuttgart",
    "Mainz": "1. FSV Mainz 05",
    "Augsburg": "FC Augsburg",
    "Heidenheim": "1. FC Heidenheim 1846",
    "UnionBerlin": "1. FC Union Berlin",
    "HerthaBerlin": "Hertha BSC",
    "Bochum": "VfL Bochum 1848",
    "Koln": "1. FC Köln",
    # Spain
    "RealMadrid": "Real Madrid",
    "Atletico": "Atlético Madrid",
    "AtleticoMadrid": "Atlético Madrid",
    "AthleticBilbao": "Athletic Club",
    "AthleticClub": "Athletic Club",
    "Betis": "Real Betis",
    "RSociedad": "Real Sociedad",
    "RealSociedad": "Real Sociedad",
    "CeltaVigo": "Celta Vigo",
    "Alaves": "Deportivo Alavés",
    "Vallecano": "Rayo Vallecano",
    "RayoVallecano": "Rayo Vallecano",
    "LasPalmas": "UD Las Palmas",
    "Leganes": "CD Leganés",
    "RealValladolid": "Real Valladolid",
    "Osasuna": "CA Osasuna",
    # Italy
    "Inter": "Inter",
    "InterMilan": "Inter",
    "ACMilan": "AC Milan",
    "Milan": "AC Milan",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Atalanta": "Atalanta BC",
    "Fiorentina": "ACF Fiorentina",
    "Torino": "Torino FC",
    "Bologna": "Bologna FC 1909",
    "Udinese": "Udinese Calcio",
    "Cagliari": "Cagliari Calcio",
    "Empoli": "Empoli FC",
    "Verona": "Hellas Verona",
    "Monza": "AC Monza",
    "Lecce": "US Lecce",
    "Parma": "Parma Calcio 1913",
    "Genoa": "Genoa CFC",
    "Sassuolo": "US Sassuolo",
    "Salernitana": "US Salernitana 1919",
    "Frosinone": "Frosinone Calcio",
    "Como": "Como 1907",
    "Venezia": "Venezia FC",
    # Portugal
    "Sporting": "Sporting CP",
    "SportingCP": "Sporting CP",
    # Netherlands
    "AZ": "AZ Alkmaar",
    "AZAlkmaar": "AZ Alkmaar",
    "PSV": "PSV Eindhoven",
    "PSVEindhoven": "PSV Eindhoven",
    "Feyenoord": "Feyenoord Rotterdam",
    "Twente": "FC Twente",
    # Turkey
    "Galatasaray": "Galatasaray SK",
    "Fenerbahce": "Fenerbahçe SK",
    "Besiktas": "Beşiktaş JK",
    # Scotland
    "Celtic": "Celtic FC",
    "Rangers": "Rangers FC",
    # Belgium
    "ClubBrugge": "Club Brugge KV",
    "Anderlecht": "RSC Anderlecht",
    # Austria
    "Salzburg": "FC Red Bull Salzburg",
    "RBSalzburg": "FC Red Bull Salzburg",
    "RapidVienna": "SK Rapid Wien",
    "Rapid": "SK Rapid Wien",
    "AustriaVienna": "FK Austria Wien",
    "Sturm": "SK Sturm Graz",
    "SturmGraz": "SK Sturm Graz",
    "LASK": "LASK",
    "Wolfsberg": "Wolfsberger AC",
    "WAC": "Wolfsberger AC",
    "Hartberg": "TSV Hartberg",
    "Altach": "SCR Altach",
    "Rheindorf": "SCR Altach",
    "Klagenfurt": "SK Austria Klagenfurt",
    "Tirol": "WSG Tirol",
    "WSGTirol": "WSG Tirol",
    "Blau-Weiss": "FC Blau-Weiß Linz",
    "BWLinz": "FC Blau-Weiß Linz",
    # Switzerland
    "YoungBoys": "BSC Young Boys",
    "Basel": "FC Basel 1893",
    "Zurich": "FC Zürich",
    "Lugano": "FC Lugano",
    "Servette": "Servette FC",
    "StGallen": "FC St. Gallen 1879",
    "Luzern": "FC Luzern",
    "Sion": "FC Sion",
    "Winterthur": "FC Winterthur",
    "GCZurich": "Grasshopper Club Zürich",
    "Grasshoppers": "Grasshopper Club Zürich",
    "Lausanne": "FC Lausanne-Sport",
    "Yverdon": "Yverdon Sport FC",
    # Greece
    "Olympiacos": "Olympiacos FC",
    "Olympiakos": "Olympiacos FC",
    "Panathinaikos": "Panathinaikos FC",
    "PAOK": "PAOK FC",
    "AEK": "AEK Athens FC",
    "AEKAthens": "AEK Athens FC",
    "Aris": "Aris Thessaloniki FC",
    "ArisThessaloniki": "Aris Thessaloniki FC",
    "OFICrete": "OFI Crete",
    "Volos": "Volos NFC",
    "Asteras": "Asteras Tripolis FC",
    "Atromitos": "Atromitos FC",
    "Panetolikos": "Panetolikos FC",
    "Lamia": "PAS Lamia 1964",
    # Czech Republic
    "SpartaPrague": "Sparta Prague",
    "SlaviaPrague": "Slavia Prague",
    "Slavia": "Slavia Prague",
    "PlzenViktoria": "Viktoria Plzeň",
    "Viktoria": "Viktoria Plzeň",
    "Plzen": "Viktoria Plzeň",
    "BanikOstrava": "FC Baník Ostrava",
    "Ostrava": "FC Baník Ostrava",
    "SlovackoBrod": "1. FC Slovácko",
    "Slovacko": "1. FC Slovácko",
    "Jablonec": "FK Jablonec",
    "Mlada": "FK Mladá Boleslav",
    "MladaBoleslav": "FK Mladá Boleslav",
    "Liberec": "FC Slovan Liberec",
    "Hradec": "FC Hradec Králové",
    "HradecKralove": "FC Hradec Králové",
    "Bohemians1905": "Bohemians 1905",
    "Bohemians": "Bohemians 1905",
    "Teplice": "FK Teplice",
    # Denmark
    "Copenhagen": "FC Copenhagen",
    "FCCopenhagen": "FC Copenhagen",
    "Midtjylland": "FC Midtjylland",
    "FCMidtjylland": "FC Midtjylland",
    "Brondby": "Brøndby IF",
    "BrondbyIF": "Brøndby IF",
    "Nordsjaelland": "FC Nordsjælland",
    "FCNordsjaelland": "FC Nordsjælland",
    "AarhusBold": "Aarhus GF",
    "AGF": "Aarhus GF",
    "Aarhus": "Aarhus GF",
    "Silkeborg": "Silkeborg IF",
    "Randers": "Randers FC",
    "Viborg": "Viborg FF",
    "Lyngby": "Lyngby BK",
    "OdenseBK": "Odense Boldklub",
    "Odense": "Odense Boldklub",
    "Aalborg": "AaB",
    "AalborgBK": "AaB",
    # Croatia
    "DinamoZagreb": "GNK Dinamo Zagreb",
    "HajdukSplit": "HNK Hajduk Split",
    "Hajduk": "HNK Hajduk Split",
    "Rijeka": "HNK Rijeka",
    "Osijek": "NK Osijek",
    "Lokomotiva": "NK Lokomotiva Zagreb",
    "Gorica": "HNK Gorica",
    "Varazdin": "NK Varaždin",
    "Istra1961": "NK Istra 1961",
    # Serbia
    "CrvenaZvezda": "FK Crvena zvezda",
    "RedStarBelgrade": "FK Crvena zvezda",
    "RedStar": "FK Crvena zvezda",
    "Partizan": "FK Partizan",
    "PartizanBelgrade": "FK Partizan",
    "Vojvodina": "FK Vojvodina",
    "Cukaricki": "FK Čukarički",
    "TSC": "FK TSC Bačka Topola",
    "Backa": "FK TSC Bačka Topola",
    # Norway
    "Bodo": "FK Bodø/Glimt",
    "BodoGlimt": "FK Bodø/Glimt",
    "Rosenborg": "Rosenborg BK",
    "Molde": "Molde FK",
    "Viking": "Viking FK",
    "Brann": "SK Brann",
    "SKBrann": "SK Brann",
    "Lillestrom": "Lillestrøm SK",
    "ValerengaOslo": "Vålerenga Fotball",
    "Valerenga": "Vålerenga Fotball",
    "Tromso": "Tromsø IL",
    "Stromsgodset": "Strømsgodset IF",
    "Sarpsborg": "Sarpsborg 08 FF",
    "HamKam": "Hamarkameratene",
    # Sweden
    "Malmo": "Malmö FF",
    "MalmoFF": "Malmö FF",
    "AIK": "AIK",
    "Djurgarden": "Djurgårdens IF",
    "DjurgardenIF": "Djurgårdens IF",
    "IFElfsborg": "IF Elfsborg",
    "Elfsborg": "IF Elfsborg",
    "Hacken": "BK Häcken",
    "BKHacken": "BK Häcken",
    "Hammarby": "Hammarby IF",
    "HammarbyIF": "Hammarby IF",
    "IFKGoteborg": "IFK Göteborg",
    "Goteborg": "IFK Göteborg",
    "Norrkoping": "IFK Norrköping",
    "IFKNorrkoping": "IFK Norrköping",
    "Sirius": "IK Sirius",
    "Kalmar": "Kalmar FF",
    "Varberg": "Varbergs BoIS FC",
    "Mjallby": "Mjällby AIF",
    # Poland
    "Legia": "Legia Warsaw",
    "LegiaWarsaw": "Legia Warsaw",
    "LechPoznan": "Lech Poznań",
    "Lech": "Lech Poznań",
    "Rakow": "Raków Częstochowa",
    "RakowCzestochowa": "Raków Częstochowa",
    "PogonSzczecin": "Pogoń Szczecin",
    "Pogon": "Pogoń Szczecin",
    "Jagiellonia": "Jagiellonia Białystok",
    "JagielloniaBialystok": "Jagiellonia Białystok",
    "Gornik": "Górnik Zabrze",
    "GornikZabrze": "Górnik Zabrze",
    "SlaAskWroclaw": "Śląsk Wrocław",
    "Slask": "Śląsk Wrocław",
    "Wisla": "Wisła Kraków",
    "WislaKrakow": "Wisła Kraków",
    "Piast": "Piast Gliwice",
    "PiastGliwice": "Piast Gliwice",
    "Cracovia": "MKS Cracovia",
    "Warta": "Warta Poznań",
    "WartaPoznan": "Warta Poznań",
    "Zaglebie": "Zagłębie Lubin",
    "ZaglebieLubin": "Zagłębie Lubin",
    # Romania
    "FCSB": "FCSB",
    "SteauaBucharest": "FCSB",
    "CFRCluj": "CFR Cluj",
    "CRaiova": "Universitatea Craiova",
    "UCraiova": "Universitatea Craiova",
    "Craiova": "Universitatea Craiova",
    "RapidBucharest": "Rapid București",
    "DinamoBucharest": "Dinamo București",
    "Sepsi": "Sepsi OSK",
    # Ukraine
    "ShakhtarDonetsk": "Shakhtar Donetsk",
    "Shakhtar": "Shakhtar Donetsk",
    "DynamoKyiv": "Dynamo Kyiv",
    "Dnipro1": "SC Dnipro-1",
    "Vorskla": "Vorskla Poltava",
    "Zorya": "Zorya Luhansk",
    "ZoryaLuhansk": "Zorya Luhansk",
    "Oleksandriya": "FC Oleksandriya",
    # Russia
    "ZenitStPetersburg": "Zenit St. Petersburg",
    "Zenit": "Zenit St. Petersburg",
    "Spartak": "Spartak Moscow",
    "SpartakMoscow": "Spartak Moscow",
    "CSKA": "CSKA Moscow",
    "CSKAMoscow": "CSKA Moscow",
    "LokomotivMoscow": "Lokomotiv Moscow",
    "Lokomotiv": "Lokomotiv Moscow",
    "Krasnodar": "FK Krasnodar",
    "Rostov": "FK Rostov",
    "Rubin": "Rubin Kazan",
    "RubinKazan": "Rubin Kazan",
    "SochiFC": "PFC Sochi",
    "Sochi": "PFC Sochi",
    "Akhmat": "Akhmat Grozny",
    # Bulgaria
    "LudogoretsRazgrad": "Ludogorets Razgrad",
    "Ludogorets": "Ludogorets Razgrad",
    "LevskiSofia": "PFC Levski Sofia",
    "Levski": "PFC Levski Sofia",
    "CSKASofia": "CSKA Sofia",
    "Botev": "Botev Plovdiv",
    "BotevPlovdiv": "Botev Plovdiv",
    "Lokomotiv1929": "Lokomotiv Plovdiv",
    "LokomotivPlovdiv": "Lokomotiv Plovdiv",
    # Hungary
    "Ferencvaros": "Ferencvárosi TC",
    "FerencvarosTC": "Ferencvárosi TC",
    "MOLFehérvár": "Fehérvár FC",
    "Fehervar": "Fehérvár FC",
    "Puskas": "Puskás Akadémia FC",
    "PuskasAkademia": "Puskás Akadémia FC",
    "Ujpest": "Újpest FC",
    "UjpestFC": "Újpest FC",
    "Kecskemeti": "Kecskeméti TE",
    "Debrecen": "Debreceni VSC",
    "DebrecenVSC": "Debreceni VSC",
    # Cyprus
    "APOEL": "APOEL Nicosia",
    "APOELNicosia": "APOEL Nicosia",
    "Omonia": "AC Omonia",
    "OmoniaNicosia": "AC Omonia",
    "AnoOrthosis": "Anorthosis Famagusta",
    "Anorthosis": "Anorthosis Famagusta",
    "AELLimassol": "AEL Limassol",
    "Apollon": "Apollon Limassol",
    "ApollonLimassol": "Apollon Limassol",
    "PaphosFC": "Pafos FC",
    "Pafos": "Pafos FC",
    # Finland
    "HJK": "HJK Helsinki",
    "HJKHelsinki": "HJK Helsinki",
    "KuPS": "Kuopion Palloseura",
    "KuopionPS": "Kuopion Palloseura",
    "IlvesT": "Tampereen Ilves",
    "Ilves": "Tampereen Ilves",
    "SJK": "SJK Seinäjoki",
    "SJKSeinajoki": "SJK Seinäjoki",
    "InterTurku": "FC Inter Turku",
    "Honka": "FC Honka",
    "Haka": "FC Haka",
    # Slovakia
    "SlovanBratislava": "ŠK Slovan Bratislava",
    "Bratislava": "ŠK Slovan Bratislava",
    "SpartakTrnava": "FC Spartak Trnava",
    "Trnava": "FC Spartak Trnava",
    "DACDunajska": "FC DAC 1904 Dunajská Streda",
    "DAC": "FC DAC 1904 Dunajská Streda",
    "Zilina": "MŠK Žilina",
    "MSKZilina": "MŠK Žilina",
    "Ruzomberok": "MFK Ružomberok",
    # Slovenia
    "Maribor": "NK Maribor",
    "NKMaribor": "NK Maribor",
    "Olimpija": "NK Olimpija Ljubljana",
    "OlimpijaLjubljana": "NK Olimpija Ljubljana",
    "Celje": "NK Celje",
    "Domzale": "NK Domžale",
    "NKDomzale": "NK Domžale",
    "Mura": "NŠ Mura",
    "NSMura": "NŠ Mura",
    "Koper": "FC Koper",
    # Bosnia and Herzegovina
    "ZeljeznicarSarajevo": "FK Željezničar Sarajevo",
    "Zeljeznicar": "FK Željezničar Sarajevo",
    "SarajevoFK": "FK Sarajevo",
    "FKSarajevo": "FK Sarajevo",
    "ZrinjskiMostar": "HŠK Zrinjski Mostar",
    "Zrinjski": "HŠK Zrinjski Mostar",
    "BoraczBanjaluka": "FK Borac Banja Luka",
    "Borac": "FK Borac Banja Luka",
    "TuzlaCity": "FK Tuzla City",
    "VelezMostar": "FK Velež Mostar",
    # Israel
    "MaccabiTelAviv": "Maccabi Tel Aviv FC",
    "MaccabiTA": "Maccabi Tel Aviv FC",
    "MaccabiHaifa": "Maccabi Haifa FC",
    "HapoeiBeerSheva": "Hapoel Be'er Sheva FC",
    "HapoelBeerSheva": "Hapoel Be'er Sheva FC",
    "BeitarJerusalem": "Beitar Jerusalem FC",
    "Beitar": "Beitar Jerusalem FC",
    "MaccabiNetanya": "Maccabi Netanya FC",
    "HapoelTelAviv": "Hapoel Tel Aviv FC",
    "HapoelTA": "Hapoel Tel Aviv FC",
    # Kazakhstan
    "Astana": "FC Astana",
    "FCAstana": "FC Astana",
    "Kairat": "FC Kairat",
    "KairatAlmaty": "FC Kairat",
    "Tobol": "FC Tobol",
    "TobolKostanay": "FC Tobol",
    "Ordabasy": "FC Ordabasy",
    "Aktobe": "FC Aktobe",
    # Iceland
    "Vikingur": "Víkingur Reykjavík",
    "VikingurReykjavik": "Víkingur Reykjavík",
    "Valur": "Valur Reykjavík",
    "Breidablik": "Breiðablik",
    "FH": "FH Hafnarfjörður",
    "FHHafnarfjordur": "FH Hafnarfjörður",
    "KR": "KR Reykjavík",
    "KRReykjavik": "KR Reykjavík",
    "Stjarnan": "Stjarnan FC",
    # Ireland
    "ShamrockRovers": "Shamrock Rovers FC",
    "Shamrock": "Shamrock Rovers FC",
    "Dundalk": "Dundalk FC",
    "DundalkFC": "Dundalk FC",
    "Bohemian": "Bohemian FC",
    "BohemianFC": "Bohemian FC",
    "StPatricksAthletic": "St Patrick's Athletic FC",
    "StPats": "St Patrick's Athletic FC",
    "Shelbourne": "Shelbourne FC",
    "ShelFC": "Shelbourne FC",
    "Derry": "Derry City FC",
    "DerryCity": "Derry City FC",
    "Drogheda": "Drogheda United FC",
    "Sligo": "Sligo Rovers FC",
    "SligoRovers": "Sligo Rovers FC",
    # Wales
    "TNS": "The New Saints FC",
    "NewSaints": "The New Saints FC",
    "Connah": "Connah's Quay Nomads FC",
    "ConnahsQuay": "Connah's Quay Nomads FC",
    "BarryTown": "Barry Town United FC",
    "Bala": "Bala Town FC",
    "BalaTown": "Bala Town FC",
    "Caernarfon": "Caernarfon Town FC",
    "Penybont": "Penybont FC",
    # Georgia
    "DinamoTbilisi": "FC Dinamo Tbilisi",
    "DinamoT": "FC Dinamo Tbilisi",
    "TorpedoKutaisi": "FC Torpedo Kutaisi",
    "Torpedo": "FC Torpedo Kutaisi",
    "DinamoBatumi": "FC Dinamo Batumi",
    "Saburtalo": "FC Saburtalo Tbilisi",
    "Dila": "FC Dila Gori",
    "DilaGori": "FC Dila Gori",
    # Portugal (expansion)
    "Porto": "FC Porto",
    "FCPorto": "FC Porto",
    "Benfica": "SL Benfica",
    "SLBenfica": "SL Benfica",
    "Braga": "SC Braga",
    "SCBraga": "SC Braga",
    "Vitoria": "Vitória SC",
    "VitoriaSC": "Vitória SC",
    "Guimaraes": "Vitória SC",
    "Famalicao": "FC Famalicão",
    "GilVicente": "Gil Vicente FC",
    "Boavista": "Boavista FC",
    "CasaPia": "Casa Pia AC",
    "Arouca": "FC Arouca",
    "RioAve": "Rio Ave FC",
    "Estoril": "Estoril Praia",
    "Moreirense": "Moreirense FC",
    "AVS": "AVS Futebol SAD",
    # Belgium (expansion)
    "Genk": "KRC Genk",
    "KRCGenk": "KRC Genk",
    "Antwerp": "Royal Antwerp FC",
    "RoyalAntwerp": "Royal Antwerp FC",
    "StandardLiege": "Standard Liège",
    "Standard": "Standard Liège",
    "Gent": "KAA Gent",
    "KAAGent": "KAA Gent",
    "UnionSG": "Royale Union Saint-Gilloise",
    "UnionStGilloise": "Royale Union Saint-Gilloise",
    "CercleBrugge": "Cercle Brugge KSV",
    "Mechelen": "KV Mechelen",
    "KVMechelen": "KV Mechelen",
    "Charleroi": "Sporting Charleroi",
    "SportingCharleroi": "Sporting Charleroi",
    "Westerlo": "KVC Westerlo",
    "Kortrijk": "KV Kortrijk",
    "OHLeuven": "OH Leuven",
    "Eupen": "KAS Eupen",
    # Netherlands (expansion)
    "Ajax": "AFC Ajax",
    "AFCAjax": "AFC Ajax",
    "Vitesse": "Vitesse",
    "Utrecht": "FC Utrecht",
    "FCUtrecht": "FC Utrecht",
    "Heerenveen": "SC Heerenveen",
    "SCHeerenveen": "SC Heerenveen",
    "Groningen": "FC Groningen",
    "NEC": "NEC Nijmegen",
    "NECNijmegen": "NEC Nijmegen",
    "SpartaRotterdam": "Sparta Rotterdam",
    "GoAheadEagles": "Go Ahead Eagles",
    "Fortuna": "Fortuna Sittard",
    "FortunaSittard": "Fortuna Sittard",
    "Heracles": "Heracles Almelo",
    "HeraclesAlmelo": "Heracles Almelo",
    "Waalwijk": "RKC Waalwijk",
    "Volendam": "FC Volendam",
    "Almere": "Almere City FC",
    "Excelsior": "Excelsior Rotterdam",
    "Willem": "Willem II",
    "WillemII": "Willem II",
    # Scotland (expansion)
    "AberdeenFC": "Aberdeen FC",
    "Aberdeen": "Aberdeen FC",
    "HeartsFC": "Heart of Midlothian FC",
    "Hearts": "Heart of Midlothian FC",
    "Hibernian": "Hibernian FC",
    "HibernianFC": "Hibernian FC",
    "Hibs": "Hibernian FC",
    "Dundee": "Dundee FC",
    "DundeeFC": "Dundee FC",
    "DundeeUtd": "Dundee United FC",
    "DundeeUnited": "Dundee United FC",
    "StMirren": "St Mirren FC",
    "StJohnstone": "St Johnstone FC",
    "Kilmarnock": "Kilmarnock FC",
    "KilmarnockFC": "Kilmarnock FC",
    "Ross": "Ross County FC",
    "RossCounty": "Ross County FC",
    "Motherwell": "Motherwell FC",
    "MotherwellFC": "Motherwell FC",
    "Livingston": "Livingston FC",
    # Turkey (expansion)
    "Trabzonspor": "Trabzonspor",
    "Basaksehir": "İstanbul Başakşehir FK",
    "IstanbulBasaksehir": "İstanbul Başakşehir FK",
    "Sivasspor": "Sivasspor",
    "Antalyaspor": "Antalyaspor",
    "Konyaspor": "Konyaspor",
    "Kasimpasa": "Kasımpaşa SK",
    "KasimpasaSK": "Kasımpaşa SK",
    "Alanyaspor": "Alanyaspor",
    "Rizespor": "Çaykur Rizespor",
    "CaykurRizespor": "Çaykur Rizespor",
    "Hatayspor": "Hatayspor",
    "Kayserispor": "Kayserispor",
    "Adana": "Adana Demirspor",
    "AdanaDemirspor": "Adana Demirspor",
    "Gaziantep": "Gaziantep FK",
    "GaziantepFK": "Gaziantep FK",
    "Eyupspor": "Eyüpspor",
    "Pendikspor": "Pendikspor",
}

# Build reverse lookup (Sofascore → ClubElo) for history queries
_SOFASCORE_TO_CLUBELO: Dict[str, str] = {v: k for k, v in _CLUBELO_TO_SOFASCORE.items()}


def _get_clubelo_sofascore_map() -> Dict[str, str]:
    """Return ClubElo→Sofascore name mapping, augmented by REEP register.

    The hardcoded ``_CLUBELO_TO_SOFASCORE`` is always included as a
    fallback.  When the REEP teams.csv is available, its
    ``key_clubelo → name`` mapping is merged in (REEP entries can
    override hardcoded ones since REEP names are more authoritative).
    """
    merged = dict(_CLUBELO_TO_SOFASCORE)  # start with hardcoded fallback
    try:
        from backend.data.reep_registry import build_clubelo_sofascore_map

        reep_map = build_clubelo_sofascore_map()
        if reep_map:
            merged.update(reep_map)
    except Exception as exc:
        _log.debug("REEP augmentation unavailable: %s", exc)
    return merged


# ── Dynamic alias generation from REEP ───────────────────────────────────────

_dynamic_aliases_cache: Optional[Dict[str, List[str]]] = None


def _build_dynamic_aliases() -> Dict[str, List[str]]:
    """Build fuzzy alias table dynamically from REEP teams.csv.

    For every team row in REEP that has multiple name columns
    (``name``, ``key_clubelo``, ``key_fbref``, ``key_transfermarkt``),
    we normalize each variant and create bidirectional alias links.

    This supplements the hardcoded ``_EXTREME_ABBREVS`` so we don't
    have to manually maintain aliases for thousands of clubs.  The REEP
    register covers ~45,000 teams globally.

    Results are cached in-process after first successful build.
    Returns empty dict on any failure (no network in tests, graceful degradation).
    """
    global _dynamic_aliases_cache
    if _dynamic_aliases_cache is not None:
        return _dynamic_aliases_cache

    try:
        from backend.data.reep_registry import get_teams_df

        df = get_teams_df()
        if df is None:
            # Don't cache failure — retry next time (e.g. network was down)
            return {}
    except Exception as exc:
        _log.debug("REEP teams unavailable for alias building: %s", exc)
        return {}

    # Columns that may contain team names across providers
    name_columns = [
        c for c in ["name", "key_clubelo", "key_fbref", "key_transfermarkt"]
        if c in df.columns
    ]
    if not name_columns:
        return {}

    aliases: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        # Collect all non-null name variants for this team
        variants: List[str] = []
        for col in name_columns:
            val = row.get(col)
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    variants.append(s)

        if len(variants) < 2:
            continue  # nothing to cross-link

        # Normalize all variants
        normed = []
        for v in variants:
            n = _normalize_team_name(v)
            if n and len(n) >= 3:  # skip trivially short
                normed.append(n)

        # Remove duplicates but preserve order
        normed = list(dict.fromkeys(normed))
        if len(normed) < 2:
            continue

        # Create bidirectional links between all pairs
        for i, norm_a in enumerate(normed):
            for j, norm_b in enumerate(normed):
                if i != j:
                    aliases.setdefault(norm_a, [])
                    if norm_b not in aliases[norm_a]:
                        aliases[norm_a].append(norm_b)

    _log.info(
        "Built %d dynamic aliases from REEP teams (%d bidirectional links)",
        len(aliases),
        sum(len(v) for v in aliases.values()),
    )
    _dynamic_aliases_cache = aliases
    return aliases


def _get_merged_aliases() -> Dict[str, List[str]]:
    """Merge hardcoded ``_EXTREME_ABBREVS`` with dynamic REEP aliases.

    Hardcoded entries take priority (they are curated for edge cases
    like PSG, ManCity, BVB that REEP may not handle perfectly).
    Dynamic aliases fill in the gaps for the thousands of teams that
    aren't manually maintained.
    """
    dynamic = _build_dynamic_aliases()
    if not dynamic:
        return _EXTREME_ABBREVS

    # Start with dynamic, then overlay hardcoded
    merged = dict(dynamic)
    for key, val_list in _EXTREME_ABBREVS.items():
        existing = merged.get(key, [])
        # Merge: keep hardcoded entries + any dynamic extras
        combined = list(val_list)
        for v in existing:
            if v not in combined:
                combined.append(v)
        merged[key] = combined

    return merged


@dataclass
class LeagueSnapshot:
    """Per-league statistics for a single day."""

    league_code: str
    league_name: str
    date: date
    mean_elo: float
    std_elo: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean_normalized: float  # 0-100 normalized mean
    team_count: int


@dataclass
class TeamRanking:
    """A single team's normalized ranking on a date."""

    team_name: str
    league_code: str
    raw_elo: float
    normalized_score: float  # 0-100
    league_mean_normalized: float
    relative_ability: float  # team - league_mean
    match_type: str = "exact"  # "exact" or "fuzzy"


def compute_daily_rankings(
    query_date: Optional[date] = None,
) -> Tuple[Dict[str, TeamRanking], Dict[str, LeagueSnapshot]]:
    """Compute global Power Rankings for all known teams on a date.

    Returns
    -------
    (team_rankings, league_snapshots)
        team_rankings: dict[team_name -> TeamRanking]
        league_snapshots: dict[league_code -> LeagueSnapshot]
    """
    if query_date is None:
        query_date = date.today()

    key = cache.make_key("power_rankings", query_date.isoformat())
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    # Step 1 — Collect all team Elo scores
    all_teams: Dict[str, Tuple[float, str]] = {}  # team -> (elo, league_code)

    # Build the ClubElo→Sofascore name map (hardcoded + REEP augmentation).
    clubelo_name_map = _get_clubelo_sofascore_map()

    # European clubs from ClubElo
    try:
        ce_df = clubelo_client.get_all_by_date(query_date)
        if ce_df is not None and len(ce_df) > 0:
            for raw_name in ce_df.index:
                elo_val = float(ce_df.loc[raw_name, "elo"])
                ce_league = ce_df.loc[raw_name, "league"]
                # Map ClubElo league to our code.  When soccerdata is used
                # it only maps 5 major leagues — the rest become NaN.
                # Fall back to our own mapping from country+level columns.
                league_code = _clubelo_to_code(ce_league)
                if league_code is None and "country" in ce_df.columns:
                    league_code = _clubelo_to_code_from_country(
                        ce_df.loc[raw_name, "country"],
                        ce_df.loc[raw_name, "level"] if "level" in ce_df.columns else 1,
                    )
                if league_code:
                    # Canonicalize ClubElo abbreviated name → Sofascore
                    # full name so that dropdown selections match directly.
                    canonical = clubelo_name_map.get(str(raw_name), str(raw_name))
                    all_teams[canonical] = (elo_val, league_code)
            _log.info("ClubElo loaded %d teams", len([t for t in all_teams]))
        else:
            _log.warning("ClubElo returned empty DataFrame for %s", query_date)
    except Exception as exc:
        _log.exception("ClubElo data fetch failed: %s", exc)

    # Non-European clubs from WorldFootballElo
    for code, info in LEAGUES.items():
        if info.clubelo_league is not None:
            continue  # already covered by ClubElo
        if info.worldelo_slug is None:
            continue
        try:
            teams = worldfootballelo_client.get_league_teams(info.worldelo_slug)
            for t in teams:
                if t.get("elo") and t.get("name"):
                    all_teams[t["name"]] = (t["elo"], code)
        except Exception as exc:
            _log.warning("WorldElo fetch failed for %s: %s", info.worldelo_slug, exc)

    if not all_teams:
        return {}, {}

    # Step 2 — Global normalization 0-100
    all_elos = [elo for elo, _ in all_teams.values()]
    global_min = min(all_elos)
    global_max = max(all_elos)

    def normalize(elo: float) -> float:
        return elo_router.normalize_elo(elo, global_min, global_max)

    # Step 3 — Build league snapshots
    league_teams: Dict[str, List[Tuple[str, float, float]]] = {}
    for team_name, (elo, code) in all_teams.items():
        norm = normalize(elo)
        league_teams.setdefault(code, []).append((team_name, elo, norm))

    league_snapshots: Dict[str, LeagueSnapshot] = {}
    for code, members in league_teams.items():
        elos = np.array([e for _, e, _ in members])
        norms = np.array([n for _, _, n in members])
        info = LEAGUES.get(code)
        league_snapshots[code] = LeagueSnapshot(
            league_code=code,
            league_name=info.name if info else code,
            date=query_date,
            mean_elo=float(np.mean(elos)),
            std_elo=float(np.std(elos)) if len(elos) > 1 else 0.0,
            p10=float(np.percentile(norms, 10)) if len(norms) > 1 else float(norms[0]),
            p25=float(np.percentile(norms, 25)) if len(norms) > 1 else float(norms[0]),
            p50=float(np.percentile(norms, 50)),
            p75=float(np.percentile(norms, 75)) if len(norms) > 1 else float(norms[0]),
            p90=float(np.percentile(norms, 90)) if len(norms) > 1 else float(norms[0]),
            mean_normalized=float(np.mean(norms)),
            team_count=len(members),
        )

    # Step 4 — Build team rankings with relative ability
    team_rankings: Dict[str, TeamRanking] = {}
    for team_name, (elo, code) in all_teams.items():
        norm = normalize(elo)
        league_mean = league_snapshots[code].mean_normalized
        team_rankings[team_name] = TeamRanking(
            team_name=team_name,
            league_code=code,
            raw_elo=elo,
            normalized_score=norm,
            league_mean_normalized=league_mean,
            relative_ability=norm - league_mean,
        )

    result = (team_rankings, league_snapshots)
    cache.set(key, result)
    return result


def get_team_ranking(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[TeamRanking]:
    """Get a single team's Power Ranking.

    Uses fuzzy matching to handle name differences between data sources.
    For example, ClubElo returns ``"RealMadrid"`` while Sofascore returns
    ``"Real Madrid"``.

    The returned ``TeamRanking.match_type`` is ``"exact"`` for direct
    hits or ``"fuzzy"`` when the name was resolved via similarity.
    """
    teams, _ = compute_daily_rankings(query_date)

    if not teams:
        _log.warning(
            "Power Rankings empty — no teams loaded from any Elo source"
        )
        return None

    # 1. Exact match (cheapest check)
    if team_name in teams:
        ranking = teams[team_name]
        ranking.match_type = "exact"
        return ranking

    # 2. Accent-normalized exact match — handles "Atlético Madrid" matching
    #    "Atletico Madrid" or vice-versa without needing a fuzzy pass.
    norm_query = _strip_accents(team_name)
    if norm_query != team_name:
        for key in teams:
            if _strip_accents(key) == norm_query:
                ranking = teams[key]
                ranking.match_type = "exact"
                return ranking

    # 3. Build normalized lookup and try fuzzy match
    match = _fuzzy_find_team(team_name, teams)
    if match is not None:
        _log.info("Fuzzy matched '%s' -> '%s'", team_name, match)
        ranking = teams[match]
        ranking.match_type = "fuzzy"
        return ranking

    _log.warning(
        "No Power Ranking match for '%s' among %d teams", team_name, len(teams)
    )
    return None


def get_league_snapshot(
    league_code: str,
    query_date: Optional[date] = None,
) -> Optional[LeagueSnapshot]:
    """Get a single league's snapshot."""
    _, leagues = compute_daily_rankings(query_date)
    return leagues.get(league_code)


def get_relative_ability(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Return team_normalized - league_mean_normalized."""
    ranking = get_team_ranking(team_name, query_date)
    if ranking is None:
        return None
    return ranking.relative_ability


def get_change_in_relative_ability(
    team_from: str,
    team_to: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Compute change in relative ability for a transfer.

    Returns target_relative_ability - source_relative_ability.
    """
    ra_from = get_relative_ability(team_from, query_date)
    ra_to = get_relative_ability(team_to, query_date)
    if ra_from is None or ra_to is None:
        return None
    return ra_to - ra_from


def get_historical_rankings(
    team_name: str,
    months: int = 6,
) -> List[Tuple[date, float]]:
    """Return a team's normalized Power Ranking over the past *months* months.

    Returns a list of ``(date, normalized_score)`` tuples, one per month,
    oldest first.  Falls back to the current score for months where data
    is unavailable.
    """
    from datetime import timedelta

    today = date.today()
    history: List[Tuple[date, float]] = []

    for i in range(months, -1, -1):
        # Approximate month offsets (30-day intervals)
        query_date = today - timedelta(days=30 * i)
        ranking = get_team_ranking(team_name, query_date)
        if ranking is not None:
            history.append((query_date, ranking.normalized_score))

    return history


def compare_leagues(
    league_codes: List[str],
    query_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Compare multiple leagues by their Power Ranking statistics.

    Parameters
    ----------
    league_codes : list[str]
        List of league codes (e.g. ``["ENG1", "ESP1", "GER1"]``).
    query_date : date, optional
        Defaults to today.

    Returns
    -------
    list[dict] — sorted by ``mean_normalized`` descending.  Each dict:
      ``code``, ``name``, ``mean_normalized``, ``std_elo``, ``team_count``,
      ``p10``, ``p25``, ``p50``, ``p75``, ``p90``.
    """
    _, snapshots = compute_daily_rankings(query_date)
    result = []
    for code in league_codes:
        snap = snapshots.get(code)
        if snap is None:
            continue
        result.append({
            "code": code,
            "name": snap.league_name,
            "mean_normalized": round(snap.mean_normalized, 1),
            "std_elo": round(snap.std_elo, 1),
            "team_count": snap.team_count,
            "p10": round(snap.p10, 1),
            "p25": round(snap.p25, 1),
            "p50": round(snap.p50, 1),
            "p75": round(snap.p75, 1),
            "p90": round(snap.p90, 1),
        })
    result.sort(key=lambda x: x["mean_normalized"], reverse=True)
    return result


# ── Internal ─────────────────────────────────────────────────────────────────

def _strip_accents(name: str) -> str:
    """Remove diacritics from *name* while preserving case and spacing."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _clubelo_to_code(clubelo_league: Optional[str]) -> Optional[str]:
    """Map a ClubElo league string to our league code."""
    if clubelo_league is None:
        return None
    if isinstance(clubelo_league, float):  # NaN from soccerdata
        return None
    clubelo_league = str(clubelo_league)
    for code, info in LEAGUES.items():
        if info.clubelo_league == clubelo_league:
            return code
    # Unknown European league — still track it with the raw string
    return None


# Country+level → league code mapping for the soccerdata path where
# ``_translate_league`` only handles the Big-5 and sets everything else
# to NaN.  Uses the same country codes that ClubElo CSV data contains.
_COUNTRY_LEVEL_TO_CODE: Dict[str, Dict[int, str]] = {}
for _code, _info in LEAGUES.items():
    if _info.clubelo_league is not None:
        # Extract country prefix from e.g. "ENG-Premier League" → "ENG"
        _country = _info.clubelo_league.split("-")[0]
        # Derive level from league code suffix (e.g. "ENG1" → 1, "ENG2" → 2)
        _level = int(_code[-1]) if _code[-1].isdigit() else 1
        _COUNTRY_LEVEL_TO_CODE.setdefault(_country, {})[_level] = _code


def _clubelo_to_code_from_country(
    country: Optional[str],
    level: Optional[int] = 1,
) -> Optional[str]:
    """Derive league code from raw ClubElo country and level columns.

    This is a fallback for when ``soccerdata``'s ``_translate_league``
    sets the league column to NaN for non-Big-5 leagues.
    """
    if country is None or (isinstance(country, float) and pd.isna(country)):
        return None
    country = str(country).strip()
    try:
        level_int = int(level)
    except (ValueError, TypeError):
        level_int = 1
    lvl_map = _COUNTRY_LEVEL_TO_CODE.get(country, {})
    return lvl_map.get(level_int) or lvl_map.get(1)


# ── Fuzzy team name matching ─────────────────────────────────────────────────

# Only strip short abbreviation suffixes that never form the core name.
# NOTE: "Club" removed — stripping it turns "Athletic Club" into just
# "athletic" which loses identity.  "AC" is kept because "AC Milan" → "Milan"
# is unambiguous (there is no other "AC" team that conflicts).
_STRIP_ABBREVS = re.compile(
    r"\b(FC|CF|SC|AC|AS|SS|SK|FK|BK|IF|SV|VfB|VfL|TSG|BSC|"
    r"1\.\s*FC|Calcio|Futbol)\b",
    re.IGNORECASE,
)

# Minimum similarity ratio (0-1) for SequenceMatcher to accept a match.
# 0.85 prevents false positives like "Brentford" matching "AFC Bournemouth"
# or "Sivasspor" matching "Samsunspor".  Edge cases below 0.85
# (ManCity ↔ ManchesterCity = 0.667, ManUtd ↔ ManchesterUnited = 0.545)
# are handled by _EXTREME_ABBREVS instead.
_FUZZY_THRESHOLD = 0.85

# Abbreviation map for cases where SequenceMatcher mathematically fails
# (ratio < 0.5) and substring matching can't help.  These cover common
# ClubElo ↔ Sofascore naming discrepancies.
_EXTREME_ABBREVS: Dict[str, List[str]] = {
    # England
    "manchesterunited": ["manutd", "manunited", "manu"],
    "manutd": ["manchesterunited"],
    "manunited": ["manchesterunited"],
    "manchestercity": ["mancity"],
    "mancity": ["manchestercity"],
    "wolverhamptonwanderers": ["wolves", "wolverhampton"],
    "wolves": ["wolverhamptonwanderers", "wolverhampton"],
    "wolverhampton": ["wolverhamptonwanderers", "wolves"],
    "nottinghamforest": ["nottmforest", "nforest"],
    "nottmforest": ["nottinghamforest"],
    "sheffieldunited": ["sheffutd", "sheffieldutd"],
    "sheffutd": ["sheffieldunited"],
    "westhamunited": ["westham"],
    "westham": ["westhamunited"],
    "tottenhamhotspur": ["tottenham", "spurs"],
    "tottenham": ["tottenhamhotspur"],
    "spurs": ["tottenhamhotspur"],
    "newcastleunited": ["newcastle"],
    "newcastle": ["newcastleunited"],
    "brightonhovealbion": ["brighton"],
    "brighton": ["brightonhovealbion"],
    "leicestercity": ["leicester"],
    "leicester": ["leicestercity"],
    # France
    "parissaintgermain": ["psg", "parissg", "parissggermain"],
    "psg": ["parissaintgermain", "parissg"],
    "parissg": ["parissaintgermain", "psg"],
    "olympiquelyonnais": ["lyon", "olympiquelyon"],
    "lyon": ["olympiquelyonnais"],
    "olympiquedemarseille": ["marseille", "olympiquemarseille", "om"],
    "marseille": ["olympiquedemarseille"],
    "staderennais": ["rennes"],
    "rennes": ["staderennais"],
    "asmonaco": ["monaco"],
    "monaco": ["asmonaco"],
    "lilleosc": ["lille"],
    "lille": ["lilleosc"],
    "rclens": ["lens"],
    "lens": ["rclens"],
    "ogcnice": ["nice"],
    "nice": ["ogcnice"],
    # Germany
    "borussiadortmund": ["dortmund", "bvb", "bvbdortmund"],
    "dortmund": ["borussiadortmund"],
    "bvb": ["borussiadortmund"],
    "bvbdortmund": ["borussiadortmund"],
    "bayernmunich": ["bayernmunchen", "bayern"],
    "bayernmunchen": ["bayernmunich", "bayern"],
    "bayern": ["bayernmunchen", "bayernmunich"],
    "borussiamonchengladbach": ["gladbach", "borussiamgladbach", "monchengladbach"],
    "borussiamgladbach": ["gladbach", "borussiamonchengladbach", "monchengladbach"],
    "gladbach": ["borussiamonchengladbach", "borussiamgladbach"],
    "monchengladbach": ["borussiamonchengladbach", "borussiamgladbach"],
    "bayerleverkusen": ["leverkusen", "bayer04leverkusen"],
    "bayer04leverkusen": ["leverkusen", "bayerleverkusen"],
    "leverkusen": ["bayerleverkusen", "bayer04leverkusen"],
    "rbleipzig": ["leipzig"],
    "leipzig": ["rbleipzig"],
    "eintrachtfrankfurt": ["frankfurt", "efrankfurt"],
    "frankfurt": ["eintrachtfrankfurt"],
    "vflwolfsburg": ["wolfsburg"],
    "wolfsburg": ["vflwolfsburg"],
    "scfreiburg": ["freiburg"],
    "freiburg": ["scfreiburg"],
    "vfbstuttgart": ["stuttgart"],
    "stuttgart": ["vfbstuttgart"],
    "tsg1899hoffenheim": ["hoffenheim"],
    "hoffenheim": ["tsg1899hoffenheim"],
    "1fcunionberlin": ["unionberlin"],
    "unionberlin": ["1fcunionberlin"],
    "1fsvmainz05": ["mainz", "mainz05"],
    "mainz": ["mainz05"],
    "1fckoln": ["koln"],
    "koln": ["1fckoln"],
    # Spain
    "atleticomadrid": ["atletico", "atleticodemadrid", "atlmadrid"],
    "atletico": ["atleticomadrid"],
    "atlmadrid": ["atleticomadrid"],
    "athleticclub": ["athleticbilbao", "athletic", "bilbao"],
    "athleticbilbao": ["athleticclub", "athletic", "bilbao"],
    "athletic": ["athleticclub", "athleticbilbao"],
    "realbetis": ["betis", "realbetisbalompie"],
    "betis": ["realbetis"],
    "realsociedad": ["rsociedad", "lasociedad"],
    "deportivoalaves": ["alaves"],
    "alaves": ["deportivoalaves"],
    "celtavigo": ["celta", "rceltadevigo"],
    "celta": ["celtavigo"],
    "rayovallecano": ["rayo"],
    "rayo": ["rayovallecano"],
    # Italy
    "internazionale": ["inter", "intermilan", "intermilanfc", "internazionalemilano"],
    "internazionalemilano": ["inter", "intermilan", "internazionale"],
    "inter": ["internazionale", "intermilan", "internazionalemilano"],
    "intermilan": ["internazionale", "inter", "internazionalemilano"],
    "acmilan": ["milan"],
    "milan": ["acmilan"],
    "sscnapoli": ["napoli"],
    "napoli": ["sscnapoli"],
    "asroma": ["roma"],
    "roma": ["asroma"],
    "sslazio": ["lazio"],
    "lazio": ["sslazio"],
    "atalantabc": ["atalanta"],
    "atalanta": ["atalantabc"],
    "acffiorentina": ["fiorentina"],
    "fiorentina": ["acffiorentina"],
    "hellasverona": ["verona"],
    "verona": ["hellasverona"],
    # Portugal
    "sportinglisbon": ["sportingcp", "sporting"],
    "sportingcp": ["sportinglisbon", "sporting"],
    "sporting": ["sportinglisbon", "sportingcp"],
    # Netherlands
    "azmaalkmaar": ["az", "azalkmaar"],
    "azalkmaar": ["az", "azmaalkmaar"],
    "az": ["azalkmaar", "azmaalkmaar"],
    "psv": ["psveindhoven"],
    "psveindhoven": ["psv"],
    # Turkey
    "galatasaray": ["galatasaraysk"],
    "galatasaraysk": ["galatasaray"],
    "fenerbahce": ["fenerbahcesk"],
    "fenerbahcesk": ["fenerbahce"],
    "besiktas": ["besiktasjk"],
    "besiktasjk": ["besiktas"],
    # Belgium
    "clubbrugge": ["clubbruggekv"],
    "clubbruggekv": ["clubbrugge"],
    "rscanderlecht": ["anderlecht"],
    "anderlecht": ["rscanderlecht"],
    # Scotland
    "celticfc": ["celtic"],
    "celtic": ["celticfc"],
    "rangersfc": ["rangers"],
    "rangers": ["rangersfc"],
    # Austria
    "fcredbullsalzburg": ["salzburg", "rbsalzburg", "redbullsalzburg"],
    "salzburg": ["fcredbullsalzburg", "rbsalzburg"],
    "rbsalzburg": ["fcredbullsalzburg", "salzburg"],
    # South America
    "flamengo": ["flamengobj", "crflamengo"],
    "palmeiras": ["sepalmeiras"],
    "corinthians": ["sccorinthians", "corinthianspaulista"],
    "saopaulfc": ["saopaulo"],
    "saopaulo": ["saopaulfc"],
    "atleticomineiro": ["atleticomg", "camatleticomineiro"],
    "riverplate": ["cariverplate"],
    "bocajuniors": ["cabocajuniors"],
    # MLS — prevent false positives with European "City" / "United" teams
    "orlandocity": ["orlandocitysc", "orlando"],
    "newyorkcity": ["newyorkcityfc", "nycfc", "nyc"],
    "nycfc": ["newyorkcity", "newyorkcityfc"],
    "intermiami": ["intermiamifc", "intermiamicf"],
    "intermiamifc": ["intermiami"],
    "intermiamicf": ["intermiami"],
    "losangelesfc": ["lafc"],
    "lafc": ["losangelesfc"],
    "losangelesgalaxy": ["lagalaxy", "galaxy"],
    "lagalaxy": ["losangelesgalaxy"],
    "galaxy": ["losangelesgalaxy"],
    "atlantaunited": ["atlantaunitedfc", "atlanta"],
    "atlantaunitedfc": ["atlantaunited"],
    "dcunited": ["dcunitedfc"],
    "dcunitedfc": ["dcunited"],
    "portlandtimbers": ["portland", "timbers"],
    "seattlesounders": ["seattle", "sounders", "seattlesoundersfc"],
    "seattlesoundersfc": ["seattlesounders"],
    "cincinnati": ["fccincinnati"],
    "fccincinnati": ["cincinnati"],
    "nashvillesc": ["nashville"],
    "charlottefc": ["charlotte"],
    "columbusc": ["columbuscrew", "crew"],
    "columbuscrew": ["columbusc", "crew"],
    "minnesotaunited": ["minnesota"],
    "newyorkredbulls": ["nyredbulls", "redbulls", "nyrb"],
    "redbulls": ["newyorkredbulls"],
    "sportingkansascity": ["sportingkc", "kansascity"],
    "sportingkc": ["sportingkansascity"],
    "houstondy": ["houstondynamo"],
    "houstondynamo": ["houstondy", "dynamo"],
    # Saudi Arabia
    "alhilal": ["alhilalfc", "hilal"],
    "alahli": ["alahlifc", "ahli"],
    "alnassr": ["alnassrfc", "nassr"],
    "alittihad": ["alittihadfc", "ittihad"],
    # Japan
    "urawareddiamonds": ["urawareds", "urawa"],
    "yokohamamarinos": ["yokohamafmarinos", "marinos"],
    "visselkobe": ["kobe", "vissel"],
    "kawasakifrontale": ["kawasaki"],
    # Portugal
    "fcporto": ["porto"],
    "porto": ["fcporto"],
    "slbenfica": ["benfica"],
    "benfica": ["slbenfica"],
    "scbraga": ["braga"],
    "braga": ["scbraga"],
    "vitoriasc": ["vitoria", "guimaraes", "vitoriaguimaraes"],
    "vitoria": ["vitoriasc"],
    "guimaraes": ["vitoriasc"],
    # Belgium (expanded)
    "krcgenk": ["genk"],
    "genk": ["krcgenk"],
    "royalantwerpfc": ["antwerp", "royalantwerp"],
    "antwerp": ["royalantwerpfc"],
    "standardliege": ["standard"],
    "standard": ["standardliege"],
    "kaagent": ["gent"],
    "gent": ["kaagent"],
    "royaleunionstgilloise": ["unionsg", "unionstgilloise"],
    "unionsg": ["royaleunionstgilloise"],
    "unionstgilloise": ["royaleunionstgilloise"],
    "sportingcharleroi": ["charleroi"],
    "charleroi": ["sportingcharleroi"],
    # Netherlands (expanded)
    "afcajax": ["ajax"],
    "ajax": ["afcajax"],
    "fcutrecht": ["utrecht"],
    "utrecht": ["fcutrecht"],
    "scheerenveen": ["heerenveen"],
    "heerenveen": ["scheerenveen"],
    "necnijmegen": ["nec"],
    "nec": ["necnijmegen"],
    "spartarotterdam": ["spartardam"],
    "fortunasittard": ["fortuna"],
    "fortuna": ["fortunasittard"],
    "heraclesalmelo": ["heracles"],
    "heracles": ["heraclesalmelo"],
    # Scotland (expanded)
    "heartofmidlothianfc": ["hearts", "heartsfc"],
    "hearts": ["heartofmidlothianfc"],
    "hibernianfc": ["hibs", "hibernian"],
    "hibernian": ["hibernianfc"],
    "hibs": ["hibernianfc"],
    "dundeeunitedfc": ["dundeeutd", "dundeeunited"],
    "dundeeutd": ["dundeeunitedfc"],
    "dundeeunited": ["dundeeunitedfc"],
    "stmirrenfc": ["stmirren"],
    "stmirren": ["stmirrenfc"],
    "stjohnstonefc": ["stjohnstone"],
    "stjohnstone": ["stjohnstonefc"],
    "kilmarnockfc": ["kilmarnock"],
    "kilmarnock": ["kilmarnockfc"],
    "rosscountyfc": ["rosscounty", "ross"],
    "rosscounty": ["rosscountyfc"],
    "motherwellfc": ["motherwell"],
    "motherwell": ["motherwellfc"],
    # Turkey (expanded)
    "istanbulbasaksehirfk": ["basaksehir", "istanbulbasaksehir"],
    "basaksehir": ["istanbulbasaksehirfk"],
    "kasimpasask": ["kasimpasa"],
    "kasimpasa": ["kasimpasask"],
    "caykurrizespor": ["rizespor"],
    "rizespor": ["caykurrizespor"],
    "adanademirspor": ["adana"],
    "adana": ["adanademirspor"],
    "gaziantepfk": ["gaziantep"],
    "gaziantep": ["gaziantepfk"],
    # Switzerland
    "bscyoungboys": ["youngboys"],
    "youngboys": ["bscyoungboys"],
    "fcbasel1893": ["basel"],
    "basel": ["fcbasel1893"],
    "fczurich": ["zurich"],
    "zurich": ["fczurich"],
    "grasshopperclubzurich": ["grasshoppers", "gczurich"],
    "grasshoppers": ["grasshopperclubzurich"],
    "gczurich": ["grasshopperclubzurich"],
    "fcstgallen1879": ["stgallen"],
    "stgallen": ["fcstgallen1879"],
    "fclausannesport": ["lausanne"],
    "lausanne": ["fclausannesport"],
    # Greece
    "olympiacosfc": ["olympiacos", "olympiakos"],
    "olympiacos": ["olympiacosfc"],
    "olympiakos": ["olympiacosfc"],
    "panathinaikosfc": ["panathinaikos"],
    "panathinaikos": ["panathinaikosfc"],
    "paokfc": ["paok"],
    "paok": ["paokfc"],
    "aekathensfc": ["aek", "aekathens"],
    "aek": ["aekathensfc"],
    "aekathens": ["aekathensfc"],
    "aristhessalonikifc": ["aris", "aristhessaloniki"],
    "aris": ["aristhessalonikifc"],
    # Czech Republic
    "spartaprague": ["spartaprag"],
    "slaviaprague": ["slavia"],
    "slavia": ["slaviaprague"],
    "viktoriaplzen": ["plzen", "viktoria"],
    "plzen": ["viktoriaplzen"],
    "fcbanikostravafc": ["ostrava", "banikostravafc"],
    "ostrava": ["banikostravafc"],
    "fkmladaboleslav": ["mlada", "mladaboleslav"],
    "mlada": ["fkmladaboleslav"],
    "fcslovanliberec": ["liberec"],
    "liberec": ["fcslovanliberec"],
    # Denmark
    "fccopenhagen": ["copenhagen"],
    "copenhagen": ["fccopenhagen"],
    "fcmidtjylland": ["midtjylland"],
    "midtjylland": ["fcmidtjylland"],
    "brondbyif": ["brondby"],
    "brondby": ["brondbyif"],
    "fcnordsjaelland": ["nordsjaelland"],
    "nordsjaelland": ["fcnordsjaelland"],
    "aarhusgf": ["agf", "aarhus"],
    "agf": ["aarhusgf"],
    "aarhus": ["aarhusgf"],
    # Croatia
    "gnkdinamozagreb": ["dinamozagreb"],
    "dinamozagreb": ["gnkdinamozagreb"],
    "hnkhajduksplit": ["hajduk", "hajduksplit"],
    "hajduk": ["hnkhajduksplit"],
    "hajduksplit": ["hnkhajduksplit"],
    "hnkrijeka": ["rijeka"],
    "rijeka": ["hnkrijeka"],
    "nkosijek": ["osijek"],
    "osijek": ["nkosijek"],
    # Serbia
    "fkcrvenazvedza": ["redstarbelgrade", "redstar", "crvenazvedza"],
    "redstarbelgrade": ["fkcrvenazvedza"],
    "redstar": ["fkcrvenazvedza"],
    "crvenazvedza": ["fkcrvenazvedza"],
    "fkpartizan": ["partizan", "partizanbelgrade"],
    "partizan": ["fkpartizan"],
    "partizanbelgrade": ["fkpartizan"],
    "fkvojvodina": ["vojvodina"],
    "vojvodina": ["fkvojvodina"],
    # Norway
    "fkbodoglimt": ["bodo", "bodoglimt"],
    "bodo": ["fkbodoglimt"],
    "bodoglimt": ["fkbodoglimt"],
    "rosenborgbk": ["rosenborg"],
    "rosenborg": ["rosenborgbk"],
    "moldefk": ["molde"],
    "molde": ["moldefk"],
    "vikingfk": ["viking"],
    "viking": ["vikingfk"],
    "skbrann": ["brann"],
    "brann": ["skbrann"],
    "lillestromsk": ["lillestrom"],
    "lillestrom": ["lillestromsk"],
    "valerengafotball": ["valerenga"],
    "valerenga": ["valerengafotball"],
    "stromsgodsetif": ["stromsgodset"],
    "stromsgodset": ["stromsgodsetif"],
    # Sweden
    "malmoff": ["malmo"],
    "malmo": ["malmoff"],
    "djurgardensif": ["djurgarden"],
    "djurgarden": ["djurgardensif"],
    "ifelfsborg": ["elfsborg"],
    "elfsborg": ["ifelfsborg"],
    "bkhacken": ["hacken"],
    "hacken": ["bkhacken"],
    "hammarbyif": ["hammarby"],
    "hammarby": ["hammarbyif"],
    "ifkgoteborg": ["goteborg"],
    "goteborg": ["ifkgoteborg"],
    "ifknorrkoping": ["norrkoping"],
    "norrkoping": ["ifknorrkoping"],
    # Poland
    "legiawarsaw": ["legia"],
    "legia": ["legiawarsaw"],
    "lechpoznan": ["lech"],
    "lech": ["lechpoznan"],
    "rakowczestochowa": ["rakow"],
    "rakow": ["rakowczestochowa"],
    "pogonszczecin": ["pogon"],
    "pogon": ["pogonszczecin"],
    "jagielloniabialystok": ["jagiellonia"],
    "jagiellonia": ["jagielloniabialystok"],
    "gornikzabrze": ["gornik"],
    "gornik": ["gornikzabrze"],
    "slaskwroclaw": ["slask"],
    "slask": ["slaskwroclaw"],
    "wislakrakow": ["wisla"],
    "wisla": ["wislakrakow"],
    "piastgliwice": ["piast"],
    "piast": ["piastgliwice"],
    "zaglebielubin": ["zaglebie"],
    "zaglebie": ["zaglebielubin"],
    # Romania
    "fcsb": ["steauabucharest", "steaua"],
    "steauabucharest": ["fcsb"],
    "steaua": ["fcsb"],
    "universitateacraiova": ["craiova", "ucraiova"],
    "craiova": ["universitateacraiova"],
    "rapidbucuresti": ["rapidbucharest"],
    "rapidbucharest": ["rapidbucuresti"],
    # Ukraine
    "shakhtardonetsk": ["shakhtar"],
    "shakhtar": ["shakhtardonetsk"],
    "dynamokyiv": ["dynamokiev"],
    "dynamokiev": ["dynamokyiv"],
    "zoryaluhansk": ["zorya"],
    "zorya": ["zoryaluhansk"],
    # Russia
    "zenitstpetersburg": ["zenit"],
    "zenit": ["zenitstpetersburg"],
    "spartakmoscow": ["spartak"],
    "spartak": ["spartakmoscow"],
    "cskamoscow": ["cska"],
    "cska": ["cskamoscow"],
    "lokomotivmoscow": ["lokomotiv"],
    "rubinkazan": ["rubin"],
    "rubin": ["rubinkazan"],
    # Bulgaria
    "ludogoretsrazgrad": ["ludogorets"],
    "ludogorets": ["ludogoretsrazgrad"],
    "pfclevskisofia": ["levski", "levskisofia"],
    "levski": ["pfclevskisofia"],
    "levskisofia": ["pfclevskisofia"],
    "cskasofia": ["cska1948sofia"],
    "botevplovdiv": ["botev"],
    "botev": ["botevplovdiv"],
    # Hungary
    "ferencvarostc": ["ferencvaros"],
    "ferencvaros": ["ferencvarostc"],
    "fehervárfc": ["fehervar"],
    "fehervar": ["fehervárfc"],
    "puskasademiafc": ["puskas", "puskasakademia"],
    "puskas": ["puskasademiafc"],
    "ujpestfc": ["ujpest"],
    "ujpest": ["ujpestfc"],
    "debrecenivsc": ["debrecen"],
    "debrecen": ["debrecenivsc"],
    # Cyprus
    "apoelnicosia": ["apoel"],
    "apoel": ["apoelnicosia"],
    "acomonia": ["omonia", "omonianicosia"],
    "omonia": ["acomonia"],
    "anorthosisfamagusta": ["anorthosis"],
    "anorthosis": ["anorthosisfamagusta"],
    "apollonlimassol": ["apollon"],
    "apollon": ["apollonlimassol"],
    # Finland
    "hjkhelsinki": ["hjk"],
    "hjk": ["hjkhelsinki"],
    "kuopionpalloseura": ["kups"],
    "kups": ["kuopionpalloseura"],
    # Slovakia
    "skslovanbratislava": ["slovanbratislava", "bratislava"],
    "slovanbratislava": ["skslovanbratislava"],
    "fcspartaktrnava": ["trnava", "spartaktrnava"],
    "trnava": ["fcspartaktrnava"],
    "mskzilina": ["zilina"],
    "zilina": ["mskzilina"],
    # Slovenia
    "nkmaribor": ["maribor"],
    "maribor": ["nkmaribor"],
    "nkolimpijaljubljana": ["olimpija", "olimpijaljubljana"],
    "olimpija": ["nkolimpijaljubljana"],
    "nkdomzale": ["domzale"],
    "domzale": ["nkdomzale"],
    "nsmura": ["mura"],
    "mura": ["nsmura"],
    # Bosnia
    "fkzeljeznicarsarajevo": ["zeljeznicar"],
    "zeljeznicar": ["fkzeljeznicarsarajevo"],
    "fksarajevo": ["sarajevo"],
    "sarajevo": ["fksarajevo"],
    "hskzrinjskimostar": ["zrinjski", "zrinjskimostar"],
    "zrinjski": ["hskzrinjskimostar"],
    "fkvelezmostar": ["velezmostar"],
    "velezmostar": ["fkvelezmostar"],
    # Israel
    "maccabitelavivfc": ["maccabitelaviv", "maccabita"],
    "maccabitelaviv": ["maccabitelavivfc"],
    "maccabita": ["maccabitelavivfc"],
    "maccabihaifafc": ["maccabihaifa"],
    "maccabihaifa": ["maccabihaifafc"],
    "hapoelbeersheva": ["hapoelbeershevafc"],
    "hapoelbeershevafc": ["hapoelbeersheva"],
    "beitarjerusalemfc": ["beitar", "beitarjerusalem"],
    "beitar": ["beitarjerusalemfc"],
    "hapoeltelavivfc": ["hapoeltelaviv", "hapoelta"],
    "hapoeltelaviv": ["hapoeltelavivfc"],
    "hapoelta": ["hapoeltelavivfc"],
    # Kazakhstan
    "fcastana": ["astana"],
    "astana": ["fcastana"],
    "fckairat": ["kairat", "kairatalmaty"],
    "kairat": ["fckairat"],
    "kairatalmaty": ["fckairat"],
    "fctobol": ["tobol"],
    "tobol": ["fctobol"],
    # Iceland
    "vikingurreykjavik": ["vikingur"],
    "vikingur": ["vikingurreykjavik"],
    "valurreykjavik": ["valur"],
    "valur": ["valurreykjavik"],
    "breidablik": ["breidablikubr"],
    "fhhafnarfjordur": ["fh"],
    "fh": ["fhhafnarfjordur"],
    "krreykjavik": ["kr"],
    "kr": ["krreykjavik"],
    # Ireland
    "shamrockroversfc": ["shamrock", "shamrockrovers"],
    "shamrock": ["shamrockroversfc"],
    "shamrockrovers": ["shamrockroversfc"],
    "dundalkfc": ["dundalk"],
    "dundalk": ["dundalkfc"],
    "bohemianfc": ["bohemian"],
    "stpatricksathleticfc": ["stpats", "stpatricksathletic"],
    "stpats": ["stpatricksathleticfc"],
    "shelbournefc": ["shelbourne"],
    "shelbourne": ["shelbournefc"],
    "derrycityfc": ["derry", "derrycity"],
    "derry": ["derrycityfc"],
    "sligoroversfc": ["sligo", "sligorovers"],
    "sligo": ["sligoroversfc"],
    # Wales
    "thenewsaintsfc": ["tns", "newsaints"],
    "tns": ["thenewsaintsfc"],
    "newsaints": ["thenewsaintsfc"],
    "connahsquaynomadsfc": ["connah", "connahsquay"],
    "connah": ["connahsquaynomadsfc"],
    # Georgia
    "fcdinamotbilisi": ["dinamotbilisi"],
    "dinamotbilisi": ["fcdinamotbilisi"],
    "fctorpedokutaisi": ["torpedo", "torpedokutaisi"],
    "torpedokutaisi": ["fctorpedokutaisi"],
    "fcdinamobatumi": ["dinamobatumi"],
    "dinamobatumi": ["fcdinamobatumi"],
    # South America (expanded)
    "gremio": ["gremiofbpa", "gremioporealegrense"],
    "internacional": ["scinternacional", "internacionalpoa"],
    "santosfc": ["santos"],
    "santos": ["santosfc"],
    "fluminense": ["fluminensefc"],
    "fluminensefc": ["fluminense"],
    "vascoda": ["vascodagama", "crvascodagama"],
    "vascodagama": ["vascoda"],
    "botafogo": ["botafogofr"],
    "botafogofr": ["botafogo"],
    "independiente": ["caindependiente"],
    "caindependiente": ["independiente"],
    "racingclub": ["racingclubavellaneda"],
    "sanlorenzo": ["casanlorenzo"],
    "casanlorenzo": ["sanlorenzo"],
    "velez": ["velezsarsfield"],
    "velezsarsfield": ["velez"],
    "estudiantes": ["estudiantesdelaplata"],
    "estudiantesdelaplata": ["estudiantes"],
}


def _normalize_team_name(name: str) -> str:
    """Reduce a team name to a canonical key for matching.

    Steps:
    - NFKD-decompose accented characters (ü → u, é → e)
    - Lowercase
    - Strip short abbreviations (FC, CF, SC, etc.)
    - Remove all non-alphanumeric characters
    """
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = _STRIP_ABBREVS.sub("", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _fuzzy_find_team(
    query: str,
    teams: Dict[str, TeamRanking],
) -> Optional[str]:
    """Find the best-matching team name from *teams* for *query*.

    Uses automatic fuzzy matching that scales to any number of teams
    from any league — no hardcoded alias list (except extreme abbrevs).

    Matching strategy (in priority order):
    1. Exact normalized match (fast path)
    2. Extreme abbreviation lookup (PSG ↔ Paris Saint-Germain, etc.)
    3. Substring containment with overlap ratio guard
    4. Word-level matching (shared significant words)
    5. SequenceMatcher similarity ratio

    Returns the original key from *teams*, or None if no match.
    """
    q = _normalize_team_name(query)
    if not q:
        return None

    # Build normalised → original key map
    candidates: Dict[str, str] = {}
    for team_name in teams:
        norm = _normalize_team_name(team_name)
        if norm:
            candidates[norm] = team_name

    # 1. Exact normalised match
    if q in candidates:
        return candidates[q]

    # 2. Extreme abbreviation lookup — must come before substring to prevent
    #    false positives like "Paris FC" beating "PSG" for "Paris Saint-Germain"
    #    Uses merged aliases: hardcoded _EXTREME_ABBREVS + dynamic REEP aliases.
    merged_aliases = _get_merged_aliases()
    q_aliases = merged_aliases.get(q, [])
    for alias in q_aliases:
        if alias in candidates:
            return candidates[alias]
    # Reverse: check if any candidate has an alias matching q
    for norm, orig in candidates.items():
        for alias in merged_aliases.get(norm, []):
            if alias == q:
                return orig

    # 3. Substring containment — one name fully inside the other.
    #    Guard: require the shorter name to be at least 6 chars AND
    #    represent >= 45% of the longer name.  This prevents
    #    "paris" (5 chars, 29% of "parissaintgermain") from matching.
    best_sub: Optional[str] = None
    best_sub_len = 0
    for norm, orig in candidates.items():
        shorter, longer = (norm, q) if len(norm) <= len(q) else (q, norm)
        if len(shorter) < 6:
            continue
        if shorter not in longer:
            continue
        ratio = len(shorter) / len(longer)
        if ratio < 0.45:
            continue
        if len(shorter) > best_sub_len:
            best_sub = orig
            best_sub_len = len(shorter)
    if best_sub is not None:
        return best_sub

    # 4. Word-level matching — tokenize into "words" (alpha runs ≥ 4 chars)
    #    and check if any significant word from the query matches a word
    #    in a candidate or vice versa.  This catches "Borussia Dortmund"
    #    sharing "dortmund" with just "Dortmund" even when full-string
    #    substring fails.
    q_words = set(re.findall(r"[a-z]{4,}", q))
    if q_words:
        best_word_match: Optional[str] = None
        best_word_overlap = 0
        for norm, orig in candidates.items():
            c_words = set(re.findall(r"[a-z]{4,}", norm))
            shared = q_words & c_words
            if shared:
                # Weight by total characters in shared words
                overlap = sum(len(w) for w in shared)
                if overlap > best_word_overlap:
                    best_word_overlap = overlap
                    best_word_match = orig
        # Only accept if shared words represent meaningful overlap
        if best_word_match is not None and best_word_overlap >= 5:
            return best_word_match

    # 5. SequenceMatcher fuzzy matching — works for any team, any league
    #    Handles "ManCity" ↔ "manchestercity", "Flamengo" ↔ "Flamengo RJ",
    #    "São Paulo" ↔ "SaoPaulo", etc.
    best_match: Optional[str] = None
    best_ratio = 0.0
    for norm, orig in candidates.items():
        ratio = SequenceMatcher(None, q, norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = orig

    if best_ratio >= _FUZZY_THRESHOLD:
        # Secondary token validation: reject matches that share fewer than
        # 2 significant tokens (≥ 4 chars).  This prevents matches like
        # "Brentford" → "AFC Bournemouth" or "Sivasspor" → "Samsunspor"
        # where high character-level similarity doesn't reflect true identity.
        # Only applied when both sides have ≥ 2 tokens (multi-word names);
        # single-word names like "celtadevigo" vs "celtavigo" are handled
        # adequately by the SequenceMatcher ratio alone.
        q_tokens = set(re.findall(r"[a-z]{4,}", q))
        match_norm = _normalize_team_name(best_match)
        m_tokens = set(re.findall(r"[a-z]{4,}", match_norm))
        shared_tokens = q_tokens & m_tokens
        if len(q_tokens) >= 2 and len(m_tokens) >= 2 and len(shared_tokens) < 2:
            _log.debug(
                "Fuzzy match rejected (insufficient token overlap): "
                "'%s' -> '%s' (ratio=%.3f, shared_tokens=%s)",
                query, best_match, best_ratio, shared_tokens,
            )
            return None
        return best_match

    return None
