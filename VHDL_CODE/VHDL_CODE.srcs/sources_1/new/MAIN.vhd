----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/12/2020 06:54:35 PM
-- Design Name: 
-- Module Name: vga - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library ieee;
  use ieee.std_logic_1164.all;

  use ieee.std_logic_unsigned.all;
  use ieee.numeric_std.all;

entity vga is
    Port ( 
           clk : in STD_LOGIC;
           btn : in STD_LOGIC_VECTOR (4 downto 0); -- used for controlls
           sw : in STD_LOGIC_VECTOR (15 downto 0); -- used 0 for reset,1 for deactivate select, 15-14 for select
           cat : out STD_LOGIC_VECTOR (6 downto 0);
           an : out STD_LOGIC_VECTOR (3 downto 0);
           led : out STD_LOGIC_VECTOR (15 downto 0);
           vgaRed : out STD_LOGIC_VECTOR (3 downto 0);
           vgaBlue : out STD_LOGIC_VECTOR (3 downto 0);
           vgaGreen : out STD_LOGIC_VECTOR (3 downto 0);
           Hsync : out STD_LOGIC;
           Vsync : out STD_LOGIC);
end vga;

architecture Behavioral of vga is

component clkdiv is
    Port(
       clk_out1 : out std_logic;
       reset : in std_logic;
       clk_in1 : in std_logic;
       clockfall : out std_logic);
end component;

component mono is
  port (
    clk : in  std_logic;
    btn : in  std_logic_vector(4  downto 0);
    enable : out std_logic_vector(4 downto 0)
  );
end component;

component SSD is
  port (
    clk : in std_logic;
    digits  : in  std_logic_vector(15 downto 0);
    an  : out std_logic_vector(3  downto 0);
    cat : out std_logic_vector(6  downto 0)
  );
end component;

signal Color: std_logic_vector(11 downto 0);
signal MPG_out: std_logic_vector(4 downto 0);
signal reset: std_logic;
signal clk25MHz : std_logic;
signal clock : std_logic := '0';
signal counter25MHz : std_logic_vector(1 downto 0) := "00";
signal s_digits: std_logic_vector(15 downto 0) := x"1234";
signal select_building: integer range 0 to 3;

signal TCH : std_logic;
signal Hcount : integer range 0 to 1687;
signal Vcount : integer range 0 to 1065;

signal CoordX : integer range 0 to 13;
signal CoordY : integer range 0 to 13;

signal Select_CoordX : integer range 0 to 13;
signal Select_CoordY : integer range 0 to 13;

type matrix is array(13 downto 0, 13 downto 0) of integer;

signal Squares : matrix :=((0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          (0,0,0,0,0,0,0,0,0,0,0,0,0,0));

-- end declaration space
begin

clkdiv_instance: clkdiv port map(clk_out1=>clk25MHz,clk_in1=>clk,reset=>MPG_out(0),clockfall => clock);
MPG_instance: mono port map (clk => clk, btn => btn, enable => MPG_out);
SSD_instance: SSD port map (clk => clk, digits => s_digits, an => an, cat => cat);

reset <= sw(0);

process(clk)
begin
    if reset = '1' then
        counter25MHz <= "00";
    end if;
    
end process;

process(clk25MHz)
begin
if(reset = '1') then
    Hcount <= 0;
    TCH <= '0';
else if rising_edge(clk25MHz) then
    if(Hcount = 800) then --800
        Hcount <= 0;
        TCH <= '1';
    else
        Hcount <= Hcount + 1;
        TCH <= '0';
    end if;   
end if;
end if;
end process;

process(clk25MHz)
begin
if(reset = '1') then
   Vcount <= 0;
else if rising_edge(clk25MHz) then

    if (TCH = '1') then
        if(Vcount = 525) then --525
            Vcount <= 0;
        else
            Vcount <= Vcount + 1;
        end if;  
    end if; 
end if;
end if;
end process;

process(clk25MHz)
begin
if rising_edge(clk25MHz) then
    if(Vcount < 490 or Vcount > 492) then --Vcount < 490 or Vcount > 492
        Vsync <= '1';
    else
        Vsync <= '0';
    end if;
end if;
end process;

process(clk25MHz)
begin
if rising_edge(clk25MHz) then
    if(Hcount < 656 or Hcount > 752) then  --Hcount < 656 or Hcount > 752
        Hsync <= '1';
    else
        Hsync <= '0';
    end if;
end if;
end process;

process(clk25MHz)
begin
    if reset = '1' then
        Squares <=((0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0));
        Select_CoordY <= 0;
        Select_CoordX <= 0;
    elsif rising_edge(clk25MHz) then
        if MPG_out(4 downto 1) /= x"0" then
            if MPG_out(2) = '1' then
                if Select_CoordX = 0 then
                    Select_CoordX <= 13;
                else
                    Select_CoordX <= Select_CoordX - 1;
                end if;
            elsif MPG_out(3) = '1' then
                if Select_CoordX = 13 then
                        Select_CoordX <= 0;
                    else
                        Select_CoordX <= Select_CoordX + 1;
                end if;
            elsif MPG_out(1) = '1' then
                if Select_CoordY = 0 then
                    Select_CoordY <= 13;
                else
                    Select_CoordY <= Select_CoordY - 1;
                end if;
            elsif MPG_out(4) = '1' then
                if Select_CoordY = 13 then
                    Select_CoordY <= 0;
                else
                    Select_CoordY <= Select_CoordY + 1;
                end if;
            end if;
        elsif sw(1) = '1' then
            if Squares(Select_CoordX,Select_CoordY) /= 9 then
                Squares(Select_CoordX,Select_CoordY) <= 9;
            end if;
        end if;
    end if;
end process;

led <= sw;

process(clk25MHz)
variable R,G,B : std_logic_vector(3 downto 0);
begin
    if rising_edge(clk25MHz)then
        if(Hcount < 640) then  
            if(Vcount < 480) then  
                if (Hcount mod 46 = 0) or (Vcount mod 35 = 0) then
                     R:= "0000";
                     B:= "0000";
                     G:= "0000";
                 else
                    CoordX <= (Hcount/46) - 1;
                    CoordY <= (Vcount/35) - 1;
                    case Squares(CoordX,CoordY) is
                        when 0 => Color <= x"FFF";
                        when 1 => Color <= x"000";
                        when 2 => Color <= x"F00";
                        when 9 => Color <= x"999";
                        when others => Color <= x"FFF";
                     end case;
                     R:= Color(11 downto 8);
                     B:= Color(3 downto 0);
                     G:= Color(7 downto 4);
                 end if;
            end if;
        else
             R:= "0000";
             B:= "0000";
             G:= "0000";
        end if;
        vgaRed <= R;
        vgaBlue <= B;
        vgaGreen <= G;
    end if;
end process;

end Behavioral;
