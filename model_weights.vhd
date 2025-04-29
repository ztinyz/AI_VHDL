library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package weights_pkg is
  type conv1_weight_3_1_3_3_t is array (0 to 26) of std_logic_vector(7 downto 0);
  constant conv1_weight : conv1_weight_3_1_3_3_t := (
    x"3f", x"cc", x"da", x"43", x"38", x"02", x"33", x"4b",
    x"62", x"d9", x"bb", x"0d", x"3c", x"a6", x"fc", x"92",
    x"4d", x"19", x"bf", x"17", x"64", x"ac", x"8d", x"12",
    x"0b", x"59", x"15"
  );

  type conv1_bias_3_t is array (0 to 2) of std_logic_vector(7 downto 0);
  constant conv1_bias : conv1_bias_3_t := (
    x"e8", x"f1", x"c5"
  );

  type conv1_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_eps : conv1_activation_post_process_eps_1_t := (
    x"00"
  );

  type conv1_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_min_val : conv1_activation_post_process_min_val__t := (
    x"c5"
  );

  type conv1_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_max_val : conv1_activation_post_process_max_val__t := (
    x"f1"
  );

  type conv2_weight_6_3_3_3_t is array (0 to 161) of std_logic_vector(7 downto 0);
  constant conv2_weight : conv2_weight_6_3_3_3_t := (
    x"b8", x"49", x"20", x"31", x"10", x"f0", x"37", x"40",
    x"4e", x"08", x"1a", x"fd", x"e2", x"0e", x"21", x"1b",
    x"18", x"10", x"19", x"cf", x"05", x"c4", x"80", x"ea",
    x"e2", x"d6", x"d6", x"4a", x"57", x"39", x"00", x"fd",
    x"ea", x"ca", x"ce", x"d1", x"31", x"32", x"26", x"14",
    x"21", x"3c", x"0a", x"01", x"f3", x"f9", x"fe", x"02",
    x"f2", x"fe", x"11", x"fc", x"05", x"2a", x"15", x"0c",
    x"17", x"18", x"1f", x"15", x"23", x"31", x"21", x"04",
    x"16", x"3e", x"01", x"14", x"45", x"0e", x"21", x"3f",
    x"07", x"02", x"08", x"14", x"10", x"12", x"fa", x"fd",
    x"11", x"1d", x"1a", x"40", x"2b", x"ee", x"9f", x"23",
    x"03", x"45", x"1a", x"45", x"51", x"1f", x"42", x"45",
    x"1c", x"0a", x"07", x"18", x"ed", x"01", x"f4", x"bb",
    x"7c", x"dd", x"cf", x"aa", x"f1", x"08", x"2f", x"dc",
    x"f5", x"13", x"fa", x"29", x"02", x"2e", x"50", x"62",
    x"13", x"27", x"45", x"0e", x"1b", x"2c", x"f0", x"fc",
    x"0d", x"e9", x"e3", x"1e", x"ef", x"fa", x"2a", x"fd",
    x"eb", x"f3", x"d6", x"db", x"d1", x"d7", x"fd", x"fe",
    x"01", x"ff", x"d7", x"d0", x"f6", x"8a", x"e1", x"f1",
    x"a4", x"04", x"1d", x"10", x"fe", x"79", x"d0", x"1c",
    x"5b", x"99"
  );

  type conv2_bias_6_t is array (0 to 5) of std_logic_vector(7 downto 0);
  constant conv2_bias : conv2_bias_6_t := (
    x"f2", x"b4", x"df", x"b2", x"01", x"00"
  );

  type conv2_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_eps : conv2_activation_post_process_eps_1_t := (
    x"00"
  );

  type conv2_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_min_val : conv2_activation_post_process_min_val__t := (
    x"01"
  );

  type conv2_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_max_val : conv2_activation_post_process_max_val__t := (
    x"00"
  );

  type fc1_weight_32_54_t is array (0 to 1727) of std_logic_vector(7 downto 0);
  constant fc1_weight : fc1_weight_32_54_t := (
    x"54", x"3f", x"26", x"3b", x"0b", x"cf", x"c6", x"bb",
    x"fb", x"1a", x"5c", x"16", x"17", x"e9", x"d1", x"de",
    x"d3", x"f2", x"00", x"23", x"15", x"d7", x"e8", x"e8",
    x"18", x"14", x"23", x"c3", x"fc", x"a4", x"f6", x"e4",
    x"dd", x"ad", x"0f", x"3d", x"b3", x"50", x"28", x"32",
    x"5f", x"36", x"18", x"f6", x"be", x"d3", x"f2", x"e4",
    x"aa", x"d9", x"f9", x"f7", x"ed", x"dc", x"d0", x"fd",
    x"f0", x"e2", x"07", x"01", x"78", x"78", x"45", x"1f",
    x"f3", x"ed", x"23", x"d7", x"ce", x"46", x"a4", x"d0",
    x"fd", x"02", x"0f", x"1c", x"2b", x"32", x"e6", x"df",
    x"b5", x"0b", x"d2", x"e9", x"fe", x"fb", x"17", x"9b",
    x"e7", x"74", x"ea", x"1c", x"24", x"10", x"33", x"eb",
    x"28", x"2a", x"72", x"ee", x"05", x"05", x"ed", x"01",
    x"11", x"39", x"37", x"ea", x"bd", x"d8", x"e8", x"74",
    x"c9", x"0d", x"2f", x"9d", x"fd", x"9b", x"f3", x"1c",
    x"e5", x"05", x"21", x"04", x"23", x"07", x"ff", x"e3",
    x"fd", x"1d", x"fc", x"32", x"1d", x"1e", x"f6", x"ad",
    x"15", x"16", x"b2", x"dc", x"c7", x"02", x"ed", x"d9",
    x"a8", x"80", x"9e", x"b0", x"9f", x"bd", x"e2", x"db",
    x"0e", x"68", x"3c", x"2c", x"54", x"2b", x"1e", x"3a",
    x"37", x"24", x"d4", x"f5", x"e9", x"cd", x"e9", x"ed",
    x"08", x"0f", x"cc", x"ab", x"66", x"7a", x"ee", x"ec",
    x"03", x"fb", x"18", x"08", x"09", x"eb", x"e5", x"36",
    x"54", x"58", x"c5", x"be", x"b1", x"12", x"d4", x"fb",
    x"07", x"f1", x"d0", x"f8", x"ea", x"03", x"da", x"c6",
    x"96", x"f8", x"ff", x"07", x"31", x"0b", x"e6", x"34",
    x"24", x"13", x"fc", x"e5", x"f3", x"fb", x"1c", x"0f",
    x"65", x"95", x"d8", x"55", x"6c", x"a9", x"fb", x"1f",
    x"b4", x"d5", x"db", x"25", x"0a", x"07", x"1c", x"fd",
    x"fb", x"ea", x"15", x"01", x"ec", x"0e", x"3a", x"3d",
    x"08", x"37", x"0d", x"03", x"c4", x"c8", x"f3", x"fc",
    x"d3", x"ef", x"f1", x"ec", x"ee", x"07", x"0b", x"16",
    x"ed", x"f8", x"0a", x"e8", x"ed", x"0d", x"d9", x"f0",
    x"0c", x"f7", x"e1", x"ed", x"e0", x"cf", x"d8", x"10",
    x"3e", x"a3", x"06", x"3d", x"ab", x"a5", x"bb", x"de",
    x"43", x"f7", x"f5", x"1a", x"c3", x"0a", x"24", x"f8",
    x"ec", x"f3", x"1e", x"20", x"0c", x"2f", x"06", x"df",
    x"9b", x"11", x"e5", x"e8", x"40", x"e6", x"a5", x"fe",
    x"db", x"cd", x"d9", x"d7", x"f9", x"dc", x"e0", x"0d",
    x"07", x"1e", x"27", x"e1", x"f1", x"25", x"38", x"35",
    x"3d", x"31", x"54", x"65", x"04", x"1f", x"4e", x"51",
    x"09", x"c5", x"5d", x"57", x"9b", x"13", x"ea", x"39",
    x"f4", x"d9", x"ce", x"e0", x"20", x"cd", x"0e", x"2a",
    x"52", x"df", x"d5", x"ee", x"05", x"1e", x"e3", x"f4",
    x"fe", x"c9", x"05", x"04", x"e0", x"1c", x"25", x"13",
    x"ec", x"00", x"fb", x"17", x"05", x"0b", x"e5", x"f5",
    x"05", x"f4", x"e4", x"f1", x"24", x"1d", x"0f", x"12",
    x"1d", x"26", x"f3", x"ee", x"f7", x"36", x"18", x"f8",
    x"1b", x"e1", x"c9", x"cc", x"03", x"03", x"09", x"18",
    x"1b", x"f2", x"2a", x"04", x"23", x"11", x"fe", x"f0",
    x"36", x"10", x"d8", x"d3", x"a3", x"1c", x"26", x"35",
    x"23", x"f3", x"d3", x"d9", x"d9", x"e6", x"85", x"9a",
    x"ce", x"01", x"09", x"45", x"34", x"22", x"ff", x"1a",
    x"ec", x"df", x"d8", x"06", x"0f", x"14", x"2c", x"25",
    x"0a", x"ef", x"02", x"ef", x"f9", x"01", x"08", x"00",
    x"09", x"f7", x"f3", x"04", x"f8", x"07", x"01", x"05",
    x"f1", x"fe", x"02", x"fc", x"f2", x"ec", x"fa", x"f8",
    x"fb", x"f4", x"f0", x"eb", x"01", x"fe", x"fa", x"f7",
    x"02", x"ec", x"fa", x"fb", x"f7", x"07", x"0c", x"0e",
    x"fc", x"07", x"07", x"ef", x"f2", x"fd", x"09", x"03",
    x"02", x"f9", x"ed", x"f5", x"f9", x"f5", x"0a", x"eb",
    x"f9", x"05", x"ed", x"fd", x"03", x"fa", x"ee", x"fc",
    x"0d", x"fa", x"09", x"fe", x"07", x"e9", x"f5", x"05",
    x"fb", x"f5", x"ee", x"fb", x"00", x"08", x"01", x"fc",
    x"01", x"fb", x"e7", x"f3", x"f6", x"e7", x"ef", x"04",
    x"ff", x"00", x"eb", x"06", x"08", x"f0", x"ee", x"ee",
    x"f0", x"f9", x"eb", x"fb", x"f1", x"ea", x"03", x"01",
    x"f9", x"f4", x"eb", x"eb", x"87", x"bd", x"ea", x"31",
    x"3e", x"09", x"67", x"23", x"f4", x"e7", x"ca", x"a1",
    x"e0", x"c2", x"b5", x"e5", x"dc", x"e6", x"14", x"2c",
    x"13", x"dd", x"ff", x"ef", x"fb", x"2e", x"0e", x"cd",
    x"2c", x"1d", x"b2", x"11", x"20", x"f7", x"09", x"11",
    x"e7", x"fd", x"1f", x"3c", x"3b", x"2d", x"24", x"00",
    x"fa", x"d9", x"29", x"17", x"04", x"f8", x"f5", x"03",
    x"ff", x"fc", x"1c", x"fc", x"c1", x"bc", x"bd", x"d1",
    x"e4", x"b8", x"0f", x"0a", x"1a", x"cb", x"10", x"ea",
    x"23", x"eb", x"0d", x"4c", x"e2", x"25", x"c4", x"fb",
    x"f1", x"11", x"1b", x"28", x"dc", x"11", x"01", x"f5",
    x"eb", x"11", x"49", x"21", x"0d", x"18", x"f2", x"26",
    x"64", x"14", x"f6", x"36", x"d1", x"a8", x"da", x"f4",
    x"03", x"18", x"ee", x"fc", x"cb", x"0b", x"ea", x"65",
    x"b3", x"ba", x"dd", x"43", x"c2", x"01", x"74", x"aa",
    x"e2", x"98", x"dd", x"07", x"bd", x"ff", x"0e", x"e5",
    x"08", x"f0", x"11", x"0a", x"12", x"11", x"2b", x"09",
    x"20", x"36", x"1a", x"26", x"64", x"43", x"37", x"2c",
    x"26", x"32", x"17", x"04", x"8b", x"8d", x"e0", x"c0",
    x"dd", x"f0", x"13", x"f5", x"f3", x"1c", x"d0", x"d6",
    x"0f", x"ec", x"02", x"ec", x"f1", x"f2", x"d7", x"d2",
    x"bc", x"1f", x"f0", x"e5", x"34", x"26", x"cf", x"5e",
    x"9e", x"f9", x"dc", x"14", x"0c", x"15", x"24", x"2e",
    x"e7", x"cd", x"dd", x"32", x"15", x"14", x"e0", x"22",
    x"00", x"2e", x"3c", x"5a", x"16", x"3c", x"52", x"03",
    x"e3", x"ed", x"6e", x"52", x"86", x"97", x"81", x"7e",
    x"11", x"07", x"17", x"1c", x"f3", x"f3", x"3b", x"ff",
    x"f0", x"d3", x"04", x"18", x"0c", x"fe", x"32", x"b6",
    x"c4", x"a4", x"97", x"22", x"ae", x"e9", x"16", x"0b",
    x"eb", x"0d", x"e6", x"53", x"47", x"64", x"cb", x"e2",
    x"bf", x"27", x"13", x"10", x"eb", x"d6", x"d6", x"38",
    x"20", x"db", x"01", x"f1", x"cc", x"06", x"f1", x"31",
    x"f2", x"e6", x"16", x"e1", x"f0", x"11", x"e0", x"dd",
    x"e0", x"f0", x"08", x"0e", x"05", x"15", x"22", x"f6",
    x"fa", x"ee", x"c1", x"f0", x"f4", x"d0", x"d7", x"fd",
    x"14", x"3c", x"1d", x"d0", x"e6", x"26", x"e4", x"d0",
    x"31", x"e1", x"3f", x"e8", x"3c", x"0d", x"35", x"ef",
    x"f1", x"16", x"e9", x"37", x"bd", x"2a", x"49", x"27",
    x"1d", x"1c", x"00", x"fd", x"0b", x"03", x"b3", x"a9",
    x"91", x"0a", x"f2", x"d9", x"1a", x"e5", x"e5", x"1b",
    x"ff", x"cc", x"40", x"1b", x"08", x"13", x"50", x"68",
    x"25", x"34", x"65", x"0c", x"0d", x"30", x"eb", x"e7",
    x"f5", x"37", x"13", x"57", x"1f", x"e0", x"03", x"0a",
    x"f9", x"06", x"f9", x"24", x"f1", x"f0", x"03", x"e0",
    x"09", x"07", x"05", x"29", x"96", x"7b", x"c8", x"bd",
    x"ae", x"f6", x"f4", x"ea", x"4b", x"b8", x"b9", x"28",
    x"3b", x"3c", x"ff", x"0a", x"09", x"f0", x"7d", x"d7",
    x"b5", x"e7", x"ea", x"0f", x"18", x"15", x"38", x"f8",
    x"ec", x"bc", x"e0", x"11", x"86", x"ff", x"0d", x"d1",
    x"82", x"2b", x"aa", x"b7", x"19", x"8a", x"c1", x"14",
    x"08", x"12", x"2f", x"11", x"f0", x"e0", x"20", x"2f",
    x"16", x"27", x"16", x"5f", x"38", x"21", x"28", x"19",
    x"1f", x"1b", x"c7", x"c4", x"8d", x"e7", x"d7", x"c1",
    x"0d", x"04", x"ec", x"7a", x"2a", x"2e", x"11", x"1c",
    x"f6", x"23", x"20", x"11", x"f6", x"06", x"12", x"03",
    x"f4", x"fa", x"d7", x"d6", x"d4", x"1f", x"20", x"1b",
    x"e3", x"0c", x"fc", x"06", x"14", x"17", x"cc", x"1a",
    x"d5", x"ec", x"d2", x"9f", x"3e", x"39", x"6a", x"01",
    x"e0", x"16", x"2d", x"f9", x"04", x"14", x"02", x"d8",
    x"90", x"62", x"37", x"0a", x"0b", x"30", x"c4", x"ce",
    x"de", x"c2", x"e2", x"f9", x"27", x"16", x"18", x"05",
    x"f9", x"04", x"d8", x"fa", x"fc", x"e7", x"9e", x"d7",
    x"61", x"31", x"09", x"f3", x"c5", x"25", x"f6", x"dc",
    x"cb", x"f4", x"0e", x"ec", x"16", x"33", x"f7", x"dc",
    x"43", x"02", x"db", x"1b", x"d3", x"f5", x"4b", x"21",
    x"f5", x"09", x"3c", x"04", x"df", x"91", x"fe", x"03",
    x"a0", x"15", x"12", x"f8", x"43", x"3b", x"77", x"b4",
    x"8b", x"5a", x"ca", x"ed", x"f3", x"e3", x"f8", x"03",
    x"da", x"f5", x"f2", x"58", x"2b", x"29", x"fb", x"e5",
    x"d3", x"ae", x"5b", x"cf", x"42", x"52", x"40", x"f1",
    x"18", x"00", x"09", x"fb", x"25", x"d9", x"cc", x"c3",
    x"36", x"0d", x"3b", x"ad", x"c2", x"f4", x"1f", x"ff",
    x"e8", x"ff", x"e8", x"f2", x"d9", x"53", x"0c", x"cd",
    x"ef", x"1e", x"e7", x"d7", x"d1", x"09", x"20", x"2d",
    x"0f", x"0a", x"01", x"1d", x"1a", x"15", x"2a", x"36",
    x"11", x"ec", x"f4", x"d7", x"da", x"db", x"ae", x"67",
    x"27", x"41", x"1b", x"0b", x"0c", x"2f", x"1e", x"ec",
    x"c8", x"af", x"be", x"2d", x"3d", x"43", x"d1", x"de",
    x"d5", x"dc", x"8f", x"ef", x"29", x"cf", x"d3", x"c8",
    x"cf", x"01", x"84", x"76", x"09", x"df", x"06", x"22",
    x"2d", x"21", x"fc", x"b7", x"e5", x"fd", x"10", x"fa",
    x"08", x"ef", x"f3", x"fa", x"6d", x"44", x"10", x"9c",
    x"2d", x"e1", x"9f", x"6a", x"15", x"41", x"29", x"58",
    x"33", x"23", x"33", x"1e", x"09", x"29", x"d0", x"c9",
    x"01", x"ff", x"e7", x"01", x"04", x"00", x"09", x"31",
    x"0b", x"fc", x"24", x"02", x"f8", x"27", x"15", x"f8",
    x"28", x"0c", x"f6", x"e6", x"e2", x"e7", x"bc", x"fa",
    x"07", x"c5", x"ff", x"0a", x"29", x"2c", x"23", x"1a",
    x"1a", x"27", x"84", x"45", x"35", x"53", x"14", x"f2",
    x"41", x"17", x"01", x"54", x"2a", x"ec", x"07", x"02",
    x"d6", x"10", x"f0", x"14", x"ed", x"ff", x"e9", x"a5",
    x"ca", x"b8", x"06", x"f7", x"1d", x"e8", x"cc", x"f0",
    x"c3", x"e7", x"06", x"06", x"09", x"1b", x"37", x"46",
    x"2a", x"47", x"53", x"51", x"20", x"23", x"12", x"b4",
    x"55", x"49", x"79", x"0e", x"18", x"13", x"1e", x"20",
    x"3d", x"23", x"57", x"18", x"20", x"31", x"5d", x"57",
    x"e4", x"40", x"77", x"b4", x"ed", x"f0", x"0f", x"e8",
    x"f1", x"fb", x"14", x"20", x"19", x"fd", x"e4", x"d3",
    x"09", x"20", x"0f", x"93", x"35", x"1c", x"00", x"f8",
    x"05", x"13", x"06", x"f3", x"e2", x"e5", x"5a", x"f4",
    x"03", x"38", x"12", x"18", x"24", x"d9", x"b0", x"d9",
    x"ff", x"f9", x"e4", x"0e", x"1a", x"22", x"8e", x"ef",
    x"df", x"71", x"34", x"cc", x"1d", x"2a", x"db", x"fb",
    x"d4", x"ef", x"0c", x"0e", x"be", x"ff", x"e4", x"f4",
    x"4d", x"51", x"63", x"be", x"c7", x"eb", x"d9", x"2f",
    x"df", x"f4", x"04", x"0e", x"eb", x"e8", x"18", x"1b",
    x"10", x"cd", x"07", x"12", x"fa", x"2d", x"1c", x"17",
    x"f8", x"34", x"4c", x"10", x"f4", x"00", x"ea", x"ef",
    x"f0", x"cf", x"f2", x"10", x"e6", x"27", x"0d", x"a2",
    x"d7", x"cc", x"5c", x"33", x"03", x"bc", x"d5", x"06",
    x"16", x"c4", x"e1", x"09", x"1f", x"d9", x"fb", x"a9",
    x"09", x"0e", x"27", x"15", x"08", x"da", x"e0", x"7c",
    x"34", x"0e", x"a7", x"31", x"73", x"d4", x"d7", x"00",
    x"f0", x"b3", x"1a", x"33", x"08", x"48", x"1b", x"49",
    x"31", x"fb", x"16", x"18", x"0c", x"17", x"1e", x"fb",
    x"03", x"21", x"44", x"29", x"e7", x"61", x"3a", x"e7",
    x"95", x"30", x"04", x"25", x"0d", x"f3", x"07", x"e9",
    x"fe", x"f8", x"dc", x"f5", x"13", x"0c", x"1c", x"cb",
    x"fb", x"06", x"0b", x"2d", x"13", x"8a", x"fd", x"2a",
    x"89", x"f8", x"29", x"25", x"0c", x"08", x"4f", x"43",
    x"09", x"3d", x"24", x"fa", x"08", x"02", x"1a", x"d2",
    x"07", x"f3", x"ae", x"b6", x"e8", x"89", x"83", x"b2",
    x"ea", x"04", x"33", x"be", x"06", x"2a", x"e2", x"20",
    x"bd", x"f6", x"f2", x"f0", x"ee", x"e4", x"e3", x"db",
    x"05", x"de", x"0a", x"59", x"f9", x"12", x"ff", x"fa",
    x"16", x"37", x"f2", x"63", x"7e", x"b2", x"e8", x"ab",
    x"9d", x"10", x"09", x"b4", x"4b", x"77", x"7b", x"09",
    x"1f", x"3a", x"06", x"fc", x"1e", x"ba", x"cf", x"e5",
    x"ff", x"ca", x"d8", x"dd", x"e1", x"e0", x"39", x"f7",
    x"e2", x"4b", x"19", x"62", x"5a", x"51", x"09", x"1a",
    x"27", x"25", x"47", x"48", x"29", x"e1", x"c9", x"9e",
    x"24", x"09", x"1d", x"eb", x"0e", x"10", x"c4", x"d2",
    x"ba", x"4b", x"ec", x"a4", x"bd", x"df", x"0d", x"e1",
    x"ea", x"25", x"d8", x"ea", x"1e", x"08", x"02", x"f4",
    x"46", x"51", x"32", x"27", x"1a", x"15", x"eb", x"23",
    x"14", x"30", x"11", x"1f", x"41", x"16", x"54", x"e0",
    x"fe", x"23", x"d2", x"d9", x"da", x"7f", x"ee", x"f5",
    x"de", x"ee", x"28", x"06", x"06", x"fe", x"10", x"e0",
    x"ef", x"0c", x"1a", x"2e", x"f5", x"fa", x"cd", x"60",
    x"c2", x"97", x"30", x"28", x"2d", x"11", x"ff", x"ff",
    x"3b", x"1d", x"7e", x"cc", x"cb", x"e9", x"06", x"f9",
    x"fc", x"df", x"04", x"6d", x"34", x"f1", x"d9", x"01",
    x"24", x"32", x"32", x"12", x"fa", x"3b", x"1b", x"0c",
    x"5a", x"29", x"10", x"3d", x"08", x"ef", x"f3", x"17",
    x"f6", x"fe", x"4a", x"e9", x"f7", x"e4", x"dd", x"19",
    x"13", x"1a", x"bc", x"e2", x"d2", x"a9", x"87", x"d6",
    x"f5", x"dc", x"cd", x"d8", x"be", x"d1", x"34", x"60",
    x"37", x"24", x"36", x"39", x"fb", x"36", x"4a", x"ef",
    x"f5", x"03", x"1e", x"14", x"14", x"66", x"26", x"43"
  );

  type fc1_bias_32_t is array (0 to 31) of std_logic_vector(7 downto 0);
  constant fc1_bias : fc1_bias_32_t := (
    x"c3", x"01", x"ba", x"ae", x"da", x"e7", x"34", x"c1",
    x"f4", x"eb", x"7f", x"d7", x"67", x"a5", x"d5", x"c3",
    x"f4", x"b0", x"8f", x"e3", x"9f", x"21", x"28", x"16",
    x"93", x"a8", x"1b", x"2e", x"f1", x"ae", x"ea", x"30"
  );

  type fc1_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_eps : fc1_activation_post_process_eps_1_t := (
    x"00"
  );

  type fc1_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_min_val : fc1_activation_post_process_min_val__t := (
    x"28"
  );

  type fc1_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_max_val : fc1_activation_post_process_max_val__t := (
    x"7f"
  );

  type fc2_weight_10_32_t is array (0 to 319) of std_logic_vector(7 downto 0);
  constant fc2_weight : fc2_weight_10_32_t := (
    x"c5", x"8f", x"34", x"d2", x"30", x"24", x"32", x"a1",
    x"01", x"09", x"ec", x"c7", x"20", x"c8", x"26", x"3c",
    x"01", x"3a", x"1f", x"b1", x"3e", x"a3", x"20", x"99",
    x"0e", x"5b", x"6f", x"a3", x"14", x"8b", x"1f", x"86",
    x"d2", x"2c", x"36", x"68", x"7f", x"2a", x"23", x"49",
    x"0e", x"0b", x"5d", x"ff", x"88", x"51", x"cb", x"f3",
    x"df", x"20", x"86", x"74", x"dc", x"ac", x"cb", x"50",
    x"cd", x"a4", x"31", x"e8", x"52", x"a9", x"9f", x"12",
    x"3e", x"ca", x"b3", x"fd", x"f4", x"c6", x"e3", x"96",
    x"f1", x"03", x"b2", x"50", x"fb", x"d4", x"3e", x"d0",
    x"1f", x"c6", x"2b", x"c1", x"35", x"be", x"1c", x"24",
    x"09", x"7f", x"9b", x"16", x"0a", x"e3", x"bd", x"e7",
    x"21", x"c8", x"97", x"86", x"c1", x"c9", x"b8", x"c8",
    x"0b", x"05", x"b4", x"d8", x"be", x"cd", x"b4", x"e0",
    x"22", x"ae", x"87", x"24", x"cb", x"ff", x"13", x"36",
    x"0a", x"19", x"29", x"2e", x"26", x"c3", x"d1", x"21",
    x"b0", x"34", x"2f", x"59", x"2b", x"12", x"c7", x"16",
    x"06", x"fe", x"d2", x"fa", x"0c", x"31", x"2e", x"ad",
    x"bd", x"b9", x"51", x"d5", x"25", x"36", x"fd", x"b0",
    x"82", x"2a", x"22", x"8f", x"be", x"45", x"d1", x"1f",
    x"af", x"a6", x"b3", x"9d", x"d2", x"b5", x"b7", x"1a",
    x"05", x"f9", x"e4", x"c3", x"1c", x"21", x"ad", x"24",
    x"ca", x"1f", x"96", x"2d", x"d5", x"e1", x"0d", x"b7",
    x"0a", x"25", x"30", x"27", x"c8", x"c0", x"2f", x"d7",
    x"c1", x"81", x"2e", x"61", x"38", x"b3", x"b8", x"1d",
    x"f0", x"0d", x"4a", x"1a", x"20", x"24", x"1f", x"25",
    x"51", x"38", x"46", x"fd", x"c2", x"8f", x"8c", x"fc",
    x"d1", x"94", x"e9", x"e1", x"f2", x"cd", x"ca", x"94",
    x"40", x"31", x"16", x"b1", x"91", x"25", x"2b", x"0f",
    x"01", x"fb", x"a7", x"18", x"81", x"2f", x"b5", x"88",
    x"1d", x"84", x"68", x"c5", x"1c", x"2d", x"16", x"2b",
    x"08", x"b7", x"c9", x"e4", x"ce", x"2a", x"2f", x"23",
    x"2f", x"0c", x"db", x"9a", x"2b", x"d4", x"20", x"a4",
    x"ff", x"09", x"2a", x"1f", x"1a", x"d5", x"a3", x"20",
    x"17", x"12", x"9b", x"1d", x"90", x"32", x"ba", x"d6",
    x"01", x"0e", x"01", x"12", x"1d", x"bc", x"17", x"ec",
    x"be", x"2d", x"ff", x"29", x"09", x"20", x"0a", x"1d",
    x"0e", x"ff", x"cf", x"8d", x"0c", x"08", x"03", x"bd",
    x"1b", x"97", x"27", x"21", x"b4", x"34", x"db", x"c4",
    x"fb", x"23", x"10", x"cc", x"1b", x"23", x"21", x"23"
  );

  type fc2_bias_10_t is array (0 to 9) of std_logic_vector(7 downto 0);
  constant fc2_bias : fc2_bias_10_t := (
    x"b1", x"94", x"3d", x"12", x"c7", x"3b", x"0e", x"a3",
    x"10", x"c2"
  );

  type fc2_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_eps : fc2_activation_post_process_eps_1_t := (
    x"00"
  );

  type fc2_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_min_val : fc2_activation_post_process_min_val__t := (
    x"39"
  );

  type fc2_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_max_val : fc2_activation_post_process_max_val__t := (
    x"08"
  );

end package weights_pkg;
