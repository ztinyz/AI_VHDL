library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package weights_pkg is
  type conv1_weight_2_1_3_3_t is array (0 to 17) of std_logic_vector(7 downto 0);
  constant conv1_weight : conv1_weight_2_1_3_3_t := (
    x"73", x"76", x"5b", x"11", x"30", x"13", x"90", x"a4",
    x"f2", x"29", x"57", x"56", x"e1", x"70", x"33", x"39",
    x"89", x"29"
  );

  type conv1_bias_2_t is array (0 to 1) of std_logic_vector(7 downto 0);
  constant conv1_bias : conv1_bias_2_t := (
    x"f5", x"00"
  );

  type conv1_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_eps : conv1_activation_post_process_eps_1_t := (
    x"00"
  );

  type conv1_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_min_val : conv1_activation_post_process_min_val__t := (
    x"f5"
  );

  type conv1_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv1_activation_post_process_max_val : conv1_activation_post_process_max_val__t := (
    x"00"
  );

  type conv2_weight_5_2_3_3_t is array (0 to 89) of std_logic_vector(7 downto 0);
  constant conv2_weight : conv2_weight_5_2_3_3_t := (
    x"2c", x"0b", x"18", x"e1", x"24", x"3f", x"f1", x"fc",
    x"24", x"34", x"22", x"f8", x"08", x"e5", x"15", x"a4",
    x"d3", x"eb", x"05", x"e8", x"32", x"d9", x"de", x"17",
    x"cb", x"14", x"5a", x"10", x"20", x"f4", x"53", x"f7",
    x"97", x"51", x"a0", x"49", x"55", x"77", x"5f", x"02",
    x"0a", x"21", x"25", x"48", x"4b", x"9d", x"8d", x"c3",
    x"ea", x"cd", x"de", x"15", x"11", x"0c", x"ea", x"f4",
    x"07", x"f4", x"f9", x"11", x"0b", x"0f", x"2b", x"12",
    x"2d", x"2c", x"16", x"19", x"18", x"16", x"12", x"19",
    x"01", x"88", x"e0", x"9e", x"14", x"0d", x"06", x"0c",
    x"19", x"af", x"ea", x"13", x"cd", x"63", x"1f", x"12",
    x"eb", x"91"
  );

  type conv2_bias_5_t is array (0 to 4) of std_logic_vector(7 downto 0);
  constant conv2_bias : conv2_bias_5_t := (
    x"17", x"af", x"ea", x"eb", x"ef"
  );

  type conv2_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_eps : conv2_activation_post_process_eps_1_t := (
    x"00"
  );

  type conv2_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_min_val : conv2_activation_post_process_min_val__t := (
    x"af"
  );

  type conv2_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant conv2_activation_post_process_max_val : conv2_activation_post_process_max_val__t := (
    x"17"
  );

  type fc1_weight_32_45_t is array (0 to 1439) of std_logic_vector(7 downto 0);
  constant fc1_weight : fc1_weight_32_45_t := (
    x"fb", x"fa", x"e1", x"0f", x"eb", x"19", x"c3", x"d0",
    x"f3", x"aa", x"03", x"2d", x"25", x"1c", x"1a", x"52",
    x"24", x"22", x"5e", x"1e", x"3e", x"29", x"37", x"3a",
    x"9a", x"d7", x"b8", x"d7", x"d2", x"bc", x"14", x"30",
    x"1d", x"fb", x"be", x"17", x"1b", x"7d", x"bc", x"28",
    x"13", x"8f", x"1d", x"1d", x"30", x"10", x"ea", x"fc",
    x"d0", x"dd", x"da", x"11", x"13", x"10", x"27", x"2b",
    x"15", x"d1", x"eb", x"12", x"fd", x"fa", x"df", x"1a",
    x"46", x"1d", x"f8", x"0b", x"f0", x"f0", x"f5", x"07",
    x"e6", x"fb", x"ec", x"ca", x"ba", x"a0", x"1e", x"52",
    x"45", x"2e", x"27", x"3d", x"20", x"04", x"f3", x"23",
    x"03", x"f2", x"6c", x"7c", x"79", x"f7", x"02", x"37",
    x"f0", x"ea", x"07", x"e3", x"b1", x"85", x"78", x"fc",
    x"ee", x"df", x"0d", x"1a", x"04", x"b6", x"69", x"11",
    x"11", x"29", x"3c", x"1a", x"f4", x"11", x"f3", x"eb",
    x"1c", x"1c", x"c5", x"e6", x"eb", x"1e", x"5c", x"3b",
    x"60", x"35", x"55", x"54", x"46", x"43", x"35", x"f1",
    x"27", x"ad", x"ee", x"ec", x"01", x"e8", x"f6", x"e6",
    x"f1", x"e9", x"f9", x"ef", x"fa", x"01", x"d9", x"fa",
    x"00", x"3a", x"78", x"87", x"28", x"37", x"58", x"2d",
    x"0e", x"21", x"08", x"13", x"0f", x"09", x"04", x"f9",
    x"01", x"f9", x"0a", x"0c", x"9c", x"55", x"0b", x"fb",
    x"d7", x"f7", x"12", x"10", x"ce", x"f8", x"4b", x"f2",
    x"f1", x"09", x"fa", x"03", x"1d", x"4f", x"a3", x"a2",
    x"ed", x"e1", x"21", x"7f", x"ee", x"17", x"bc", x"26",
    x"79", x"bb", x"23", x"01", x"b6", x"50", x"f2", x"18",
    x"33", x"8d", x"26", x"f2", x"fd", x"3a", x"d2", x"e5",
    x"33", x"2c", x"13", x"fd", x"0f", x"c5", x"dc", x"00",
    x"11", x"02", x"61", x"1f", x"02", x"a3", x"51", x"1d",
    x"01", x"d5", x"16", x"fa", x"df", x"25", x"ff", x"e5",
    x"0a", x"07", x"4d", x"c1", x"49", x"23", x"a4", x"2d",
    x"33", x"f6", x"17", x"02", x"1e", x"0e", x"b0", x"09",
    x"e7", x"08", x"27", x"0a", x"a2", x"12", x"0f", x"56",
    x"f0", x"24", x"3f", x"e9", x"2f", x"3d", x"fa", x"a9",
    x"20", x"2d", x"f1", x"de", x"08", x"13", x"fb", x"69",
    x"95", x"c9", x"c9", x"f8", x"e6", x"cc", x"fe", x"fd",
    x"a0", x"cb", x"91", x"a8", x"b5", x"3b", x"f5", x"f1",
    x"0c", x"eb", x"18", x"b6", x"23", x"16", x"0c", x"e6",
    x"31", x"d1", x"22", x"18", x"62", x"ec", x"d4", x"0b",
    x"f5", x"df", x"c4", x"74", x"19", x"f6", x"c1", x"d4",
    x"20", x"01", x"25", x"0f", x"f5", x"f3", x"d1", x"ac",
    x"ab", x"df", x"c8", x"c9", x"1b", x"c9", x"ed", x"ee",
    x"a6", x"ee", x"04", x"12", x"f4", x"fc", x"d9", x"22",
    x"dc", x"e5", x"07", x"f9", x"4a", x"e2", x"19", x"8b",
    x"4a", x"18", x"4c", x"2f", x"da", x"c8", x"f9", x"0d",
    x"1e", x"02", x"e0", x"f6", x"dd", x"d7", x"0c", x"e9",
    x"ee", x"f2", x"34", x"06", x"84", x"27", x"11", x"74",
    x"46", x"57", x"26", x"14", x"4d", x"29", x"0f", x"25",
    x"3e", x"25", x"b1", x"ff", x"1f", x"15", x"0b", x"21",
    x"d5", x"8f", x"f6", x"da", x"e3", x"e7", x"08", x"ca",
    x"fb", x"04", x"11", x"fe", x"27", x"ec", x"e6", x"35",
    x"2a", x"f1", x"0c", x"3f", x"06", x"fc", x"bb", x"03",
    x"fc", x"e4", x"fc", x"00", x"8c", x"1d", x"07", x"07",
    x"b8", x"bb", x"1d", x"ac", x"bf", x"a0", x"c1", x"0b",
    x"11", x"f9", x"ef", x"09", x"e3", x"37", x"3e", x"ef",
    x"21", x"71", x"5a", x"36", x"ee", x"22", x"1d", x"bc",
    x"d9", x"ff", x"ae", x"8a", x"5a", x"37", x"e8", x"f9",
    x"0a", x"11", x"2b", x"28", x"d8", x"ed", x"1f", x"29",
    x"04", x"c5", x"37", x"2f", x"c3", x"08", x"01", x"a2",
    x"fb", x"e3", x"1c", x"16", x"c8", x"df", x"04", x"d5",
    x"2d", x"37", x"4a", x"a8", x"ad", x"ca", x"2b", x"0d",
    x"dc", x"cf", x"c4", x"d4", x"0d", x"00", x"ef", x"a9",
    x"42", x"c8", x"bf", x"45", x"ca", x"17", x"17", x"f1",
    x"15", x"ae", x"1b", x"0e", x"fb", x"67", x"f0", x"2d",
    x"45", x"f7", x"30", x"a3", x"e9", x"16", x"bc", x"ec",
    x"00", x"6a", x"e4", x"f2", x"f0", x"18", x"ea", x"1c",
    x"0f", x"b4", x"ec", x"27", x"00", x"f4", x"13", x"11",
    x"09", x"f7", x"f3", x"27", x"4a", x"2e", x"10", x"4b",
    x"ef", x"f9", x"f0", x"fc", x"fd", x"65", x"26", x"23",
    x"18", x"1c", x"01", x"15", x"ea", x"df", x"47", x"65",
    x"d0", x"33", x"02", x"e2", x"09", x"06", x"11", x"b2",
    x"05", x"24", x"ea", x"fd", x"04", x"0e", x"4b", x"d5",
    x"a8", x"85", x"87", x"dc", x"a6", x"ee", x"07", x"0e",
    x"db", x"44", x"d9", x"ef", x"3e", x"f1", x"f4", x"b9",
    x"eb", x"04", x"fb", x"1a", x"ee", x"76", x"f6", x"f7",
    x"cd", x"f0", x"02", x"7e", x"b1", x"1b", x"9d", x"a0",
    x"2b", x"51", x"a4", x"25", x"f9", x"4b", x"fb", x"f0",
    x"f8", x"07", x"07", x"5f", x"d6", x"a7", x"11", x"0e",
    x"06", x"ef", x"06", x"dc", x"e8", x"0b", x"17", x"f8",
    x"1b", x"14", x"fe", x"fd", x"0c", x"ea", x"d2", x"84",
    x"c6", x"04", x"2f", x"f2", x"f0", x"da", x"10", x"1a",
    x"2b", x"4d", x"22", x"6c", x"bc", x"b3", x"4b", x"b1",
    x"a3", x"30", x"e4", x"6f", x"1d", x"f4", x"1e", x"0d",
    x"ee", x"ee", x"08", x"f5", x"cd", x"0f", x"f7", x"09",
    x"fa", x"f2", x"f3", x"e6", x"f3", x"fc", x"61", x"2a",
    x"f3", x"2e", x"d7", x"19", x"a6", x"34", x"bb", x"c0",
    x"49", x"1a", x"e8", x"41", x"1a", x"e4", x"66", x"01",
    x"1b", x"1e", x"f4", x"de", x"e0", x"ed", x"97", x"d1",
    x"03", x"19", x"d4", x"1f", x"d1", x"1f", x"09", x"e2",
    x"1f", x"b0", x"fa", x"f9", x"b4", x"38", x"15", x"02",
    x"55", x"45", x"2e", x"0b", x"18", x"15", x"f8", x"02",
    x"0e", x"ca", x"58", x"1e", x"9c", x"23", x"02", x"12",
    x"fb", x"fb", x"7d", x"0d", x"24", x"5f", x"0c", x"13",
    x"72", x"18", x"27", x"c5", x"ed", x"f6", x"d8", x"04",
    x"e6", x"f6", x"11", x"10", x"d7", x"0b", x"0b", x"10",
    x"1a", x"07", x"2a", x"13", x"04", x"c3", x"dd", x"21",
    x"c9", x"cf", x"ec", x"e1", x"fa", x"ec", x"12", x"e9",
    x"da", x"f3", x"db", x"e6", x"00", x"96", x"ca", x"e1",
    x"00", x"6d", x"fb", x"fe", x"46", x"04", x"24", x"22",
    x"41", x"3f", x"08", x"de", x"10", x"fe", x"27", x"3f",
    x"df", x"39", x"07", x"0c", x"e7", x"03", x"14", x"bb",
    x"da", x"df", x"04", x"f9", x"f3", x"30", x"18", x"ff",
    x"19", x"09", x"ff", x"fa", x"03", x"14", x"1f", x"3a",
    x"1c", x"0c", x"7d", x"48", x"f5", x"11", x"ff", x"f7",
    x"13", x"f7", x"d0", x"77", x"3e", x"e4", x"d9", x"08",
    x"37", x"12", x"1e", x"8b", x"d0", x"e8", x"fe", x"ff",
    x"f1", x"f9", x"05", x"13", x"31", x"ff", x"13", x"f4",
    x"e2", x"0b", x"e2", x"b5", x"ea", x"e4", x"0f", x"fe",
    x"60", x"2c", x"16", x"f9", x"01", x"08", x"58", x"00",
    x"fd", x"65", x"6b", x"39", x"bf", x"c3", x"9d", x"93",
    x"cc", x"09", x"20", x"2b", x"10", x"b7", x"bc", x"26",
    x"2c", x"ff", x"dd", x"1e", x"35", x"d4", x"3b", x"2a",
    x"29", x"12", x"27", x"38", x"e6", x"43", x"7a", x"16",
    x"fd", x"e5", x"1f", x"49", x"02", x"4b", x"20", x"2c",
    x"2e", x"23", x"17", x"5e", x"30", x"28", x"17", x"ed",
    x"1f", x"e5", x"dc", x"fe", x"52", x"e5", x"df", x"e9",
    x"b4", x"a0", x"11", x"12", x"1b", x"d9", x"e6", x"05",
    x"11", x"3d", x"67", x"bb", x"fd", x"14", x"de", x"f2",
    x"fb", x"0c", x"f4", x"11", x"ca", x"bd", x"d8", x"57",
    x"49", x"5b", x"fd", x"f5", x"0f", x"0b", x"e0", x"de",
    x"13", x"3f", x"1f", x"21", x"2f", x"12", x"d0", x"f1",
    x"e3", x"ef", x"dd", x"d6", x"d2", x"e6", x"d3", x"24",
    x"f6", x"34", x"d5", x"ce", x"e9", x"3d", x"31", x"f8",
    x"0a", x"0f", x"02", x"dc", x"d8", x"f8", x"12", x"1d",
    x"e0", x"e8", x"39", x"e7", x"fa", x"e3", x"f1", x"dd",
    x"19", x"09", x"1d", x"08", x"0d", x"50", x"ed", x"e6",
    x"fa", x"98", x"35", x"4b", x"fd", x"f9", x"13", x"68",
    x"e6", x"30", x"26", x"08", x"eb", x"30", x"08", x"ce",
    x"e3", x"f6", x"ac", x"d6", x"2b", x"b3", x"c1", x"e4",
    x"18", x"ef", x"07", x"ce", x"1e", x"90", x"cb", x"d6",
    x"f5", x"4c", x"34", x"16", x"80", x"88", x"f1", x"53",
    x"01", x"ee", x"d3", x"34", x"28", x"25", x"10", x"2e",
    x"e0", x"11", x"06", x"15", x"f9", x"15", x"ac", x"d6",
    x"cc", x"22", x"ff", x"0f", x"d1", x"ed", x"fb", x"54",
    x"7a", x"d9", x"1f", x"0b", x"10", x"f5", x"e3", x"df",
    x"2e", x"12", x"e1", x"f8", x"ef", x"f6", x"19", x"2f",
    x"79", x"be", x"0d", x"1a", x"06", x"17", x"c8", x"04",
    x"b9", x"1f", x"f7", x"dc", x"d9", x"f7", x"e9", x"ff",
    x"cd", x"f1", x"f3", x"fa", x"fe", x"de", x"03", x"f8",
    x"45", x"0c", x"fa", x"7e", x"f2", x"10", x"25", x"2b",
    x"f3", x"1c", x"0a", x"27", x"0d", x"cb", x"e3", x"ec",
    x"71", x"35", x"1f", x"dd", x"e1", x"0f", x"1c", x"e3",
    x"f2", x"b6", x"4f", x"28", x"a9", x"58", x"3e", x"c6",
    x"15", x"19", x"44", x"1e", x"2b", x"0f", x"f0", x"89",
    x"0f", x"fe", x"03", x"f3", x"01", x"22", x"9e", x"b7",
    x"e2", x"02", x"04", x"05", x"01", x"ec", x"d9", x"41",
    x"24", x"02", x"db", x"cb", x"67", x"b5", x"d6", x"a7",
    x"cc", x"e3", x"07", x"0e", x"99", x"49", x"1e", x"26",
    x"0b", x"3b", x"11", x"0a", x"d9", x"e9", x"cc", x"87",
    x"b5", x"1e", x"8c", x"eb", x"1a", x"bc", x"14", x"fc",
    x"ec", x"f5", x"dd", x"f7", x"22", x"12", x"20", x"75",
    x"6e", x"2c", x"23", x"19", x"f4", x"02", x"0b", x"e4",
    x"f4", x"e9", x"b6", x"22", x"15", x"ed", x"fb", x"05",
    x"54", x"1c", x"0d", x"0b", x"e2", x"1c", x"d9", x"bf",
    x"be", x"0b", x"12", x"08", x"0c", x"d7", x"c6", x"f4",
    x"e5", x"01", x"0e", x"08", x"fd", x"d7", x"b3", x"be",
    x"5a", x"3d", x"47", x"24", x"03", x"cc", x"29", x"3c",
    x"1a", x"1e", x"13", x"2c", x"d1", x"d9", x"5c", x"d6",
    x"08", x"eb", x"ed", x"03", x"db", x"96", x"06", x"4b",
    x"e6", x"0e", x"21", x"fb", x"ec", x"df", x"cd", x"69",
    x"bb", x"e3", x"22", x"f5", x"fb", x"07", x"e5", x"fc",
    x"f8", x"37", x"f8", x"f9", x"24", x"30", x"f7", x"09",
    x"d6", x"98", x"a0", x"e6", x"ce", x"6d", x"d6", x"d2",
    x"ff", x"6b", x"6a", x"fa", x"14", x"1d", x"c9", x"d7",
    x"0f", x"e0", x"89", x"65", x"2a", x"0d", x"49", x"2f",
    x"4e", x"35", x"3a", x"eb", x"d3", x"ce", x"af", x"bf",
    x"d6", x"b1", x"98", x"e1", x"c8", x"e5", x"e7", x"e5",
    x"fd", x"f2", x"ec", x"da", x"ed", x"12", x"18", x"0f",
    x"ba", x"16", x"01", x"0e", x"0c", x"11", x"2f", x"e6",
    x"16", x"10", x"0e", x"0c", x"3f", x"33", x"f2", x"ea",
    x"ea", x"fc", x"ff", x"78", x"0a", x"b8", x"aa", x"32",
    x"01", x"4e", x"dd", x"c8", x"0f", x"fb", x"ea", x"28",
    x"f7", x"cd", x"d7", x"fe", x"e7", x"26", x"06", x"ef",
    x"95", x"29", x"ee", x"fd", x"ee", x"03", x"f6", x"00",
    x"2e", x"0c", x"f2", x"27", x"41", x"34", x"5b", x"0e",
    x"ff", x"f0", x"f4", x"01", x"ee", x"93", x"42", x"82",
    x"1b", x"15", x"ec", x"16", x"00", x"90", x"93", x"ca",
    x"3b", x"36", x"f5", x"10", x"1b", x"06", x"7d", x"ba",
    x"20", x"ab", x"dd", x"13", x"01", x"fc", x"14", x"c0",
    x"c4", x"de", x"16", x"fa", x"f8", x"33", x"00", x"0f"
  );

  type fc1_bias_32_t is array (0 to 31) of std_logic_vector(7 downto 0);
  constant fc1_bias : fc1_bias_32_t := (
    x"11", x"ad", x"8f", x"93", x"d0", x"9e", x"08", x"88",
    x"15", x"fd", x"14", x"2e", x"4c", x"0e", x"04", x"b9",
    x"e2", x"d6", x"21", x"59", x"31", x"c8", x"cd", x"81",
    x"64", x"be", x"8c", x"c0", x"69", x"c7", x"f9", x"92"
  );

  type fc1_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_eps : fc1_activation_post_process_eps_1_t := (
    x"00"
  );

  type fc1_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_min_val : fc1_activation_post_process_min_val__t := (
    x"11"
  );

  type fc1_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc1_activation_post_process_max_val : fc1_activation_post_process_max_val__t := (
    x"e0"
  );

  type fc2_weight_10_32_t is array (0 to 319) of std_logic_vector(7 downto 0);
  constant fc2_weight : fc2_weight_10_32_t := (
    x"2b", x"ca", x"29", x"14", x"37", x"a0", x"f0", x"30",
    x"89", x"33", x"fc", x"3f", x"ae", x"2a", x"21", x"ca",
    x"ff", x"1a", x"88", x"1b", x"80", x"13", x"a7", x"e2",
    x"ec", x"9c", x"eb", x"2b", x"14", x"8b", x"92", x"80",
    x"f7", x"d2", x"19", x"b6", x"43", x"18", x"9d", x"73",
    x"1f", x"2b", x"7f", x"2a", x"f2", x"23", x"d5", x"ec",
    x"03", x"89", x"31", x"13", x"1d", x"73", x"fd", x"7b",
    x"21", x"92", x"3e", x"f5", x"0f", x"68", x"f5", x"17",
    x"e2", x"37", x"fd", x"14", x"d4", x"9b", x"b2", x"98",
    x"c6", x"c9", x"e5", x"eb", x"2b", x"db", x"de", x"24",
    x"30", x"0d", x"ce", x"0a", x"c2", x"35", x"b5", x"a6",
    x"5b", x"e5", x"dd", x"38", x"11", x"dc", x"bc", x"32",
    x"c4", x"b1", x"cc", x"17", x"ae", x"d4", x"cf", x"8d",
    x"27", x"c7", x"8a", x"e3", x"2e", x"84", x"dc", x"fb",
    x"2f", x"0c", x"bc", x"d0", x"20", x"ca", x"24", x"ea",
    x"e4", x"d5", x"de", x"bc", x"25", x"e8", x"23", x"40",
    x"89", x"3c", x"f4", x"a9", x"c7", x"fa", x"1d", x"10",
    x"d4", x"32", x"31", x"2e", x"da", x"e3", x"07", x"fe",
    x"cb", x"ae", x"2e", x"8a", x"28", x"2f", x"1f", x"2f",
    x"0d", x"40", x"c5", x"48", x"e3", x"bb", x"2b", x"96",
    x"aa", x"e6", x"22", x"13", x"29", x"27", x"2d", x"e0",
    x"1e", x"c6", x"c8", x"a0", x"d4", x"f3", x"d5", x"ab",
    x"1f", x"1d", x"cc", x"c5", x"fa", x"d5", x"1f", x"1e",
    x"c9", x"02", x"00", x"b3", x"a1", x"dd", x"fe", x"9b",
    x"8f", x"49", x"ed", x"f2", x"42", x"0f", x"2f", x"2a",
    x"79", x"2c", x"3a", x"90", x"86", x"33", x"d5", x"53",
    x"c3", x"11", x"bf", x"be", x"17", x"27", x"0c", x"8b",
    x"2e", x"c9", x"4d", x"2f", x"dc", x"f9", x"5a", x"86",
    x"1b", x"c1", x"16", x"12", x"cf", x"95", x"aa", x"0b",
    x"2c", x"c1", x"cf", x"1d", x"20", x"98", x"20", x"29",
    x"2a", x"d2", x"34", x"29", x"12", x"f7", x"c9", x"3b",
    x"d9", x"29", x"e3", x"ad", x"cf", x"0d", x"2a", x"40",
    x"fa", x"df", x"d1", x"0f", x"17", x"15", x"29", x"bc",
    x"fa", x"f9", x"b2", x"bc", x"1b", x"25", x"1b", x"1e",
    x"c8", x"14", x"f1", x"04", x"f1", x"b7", x"00", x"e7",
    x"03", x"a7", x"f9", x"bc", x"20", x"c1", x"b3", x"c1",
    x"ea", x"1d", x"db", x"0a", x"0d", x"f7", x"15", x"2a",
    x"21", x"e4", x"2d", x"27", x"0c", x"d9", x"0c", x"10",
    x"d5", x"f7", x"29", x"f2", x"19", x"19", x"15", x"34",
    x"8a", x"26", x"af", x"c1", x"1d", x"d4", x"2b", x"8c"
  );

  type fc2_bias_10_t is array (0 to 9) of std_logic_vector(7 downto 0);
  constant fc2_bias : fc2_bias_10_t := (
    x"da", x"fc", x"07", x"35", x"eb", x"57", x"f2", x"76",
    x"61", x"c9"
  );

  type fc2_activation_post_process_eps_1_t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_eps : fc2_activation_post_process_eps_1_t := (
    x"00"
  );

  type fc2_activation_post_process_min_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_min_val : fc2_activation_post_process_min_val__t := (
    x"05"
  );

  type fc2_activation_post_process_max_val__t is array (0 to 0) of std_logic_vector(7 downto 0);
  constant fc2_activation_post_process_max_val : fc2_activation_post_process_max_val__t := (
    x"b7"
  );

end package weights_pkg;
