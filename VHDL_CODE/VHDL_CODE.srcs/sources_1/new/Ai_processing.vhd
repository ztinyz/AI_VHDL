----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07/06/2025 11:33:30 AM
-- Design Name: 
-- Module Name: Ai_processing - Behavioral
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


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.weights.all;

entity Ai_processing is
    Port (
        clk : in std_logic;
        reset : in std_logic;
        start : in std_logic;
        matrice: in matrix;
        predicted_digit : out std_logic_vector(3 downto 0);
        confidence : out std_logic_vector(7 downto 0);
        done : out std_logic
    );
end Ai_processing;

architecture Behavioral of Ai_processing is
    -- Input flattening signals
    type input_vector is array(0 to FC1_INPUTS-1) of signed(7 downto 0);
    signal flattened_input : input_vector;
    
    -- FC1 layer signals
    type fc1_output_vector is array(0 to FC1_OUTPUTS-1) of signed(15 downto 0);
    signal fc1_results : fc1_output_vector;
    signal fc1_activated : std_logic_vector(FC1_OUTPUTS-1 downto 0);
    
    -- FC2 layer signals (FC1 outputs -> 64 outputs)
    constant FC2_OUTPUTS : integer := 64;
    type fc2_output_vector is array(0 to FC2_OUTPUTS-1) of signed(15 downto 0);
    signal fc2_results : fc2_output_vector;
    signal fc2_activated : std_logic_vector(FC2_OUTPUTS-1 downto 0);
    
    -- FC3 layer signals (64 -> 10 outputs)
    type fc3_output_vector is array(0 to FC3_OUTPUTS-1) of signed(15 downto 0);
    signal fc3_results : fc3_output_vector;
    
    -- State machine signals
    type state_type is (IDLE, FLATTEN, COMPUTE_FC1, COMPUTE_FC2, COMPUTE_FC3, FIND_MAX, DONE_STATE);
    signal current_state : state_type;
    
    -- Computation signals
    signal compute_index : integer range 0 to 127;
    signal input_index : integer range 0 to 143;
    signal accumulator : signed(23 downto 0);
    
    -- Output processing signals
    signal max_value : signed(15 downto 0);
    signal max_index : integer range 0 to 9;
    signal search_index : integer range 0 to 9;
    
begin

    -- Process to flatten 14x14 matrix to 1D array (12x12 = 144 pixels)
    flatten_process: process(clk)
    variable flat_index : integer;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                -- Reset flattened input
                for i in 0 to FC1_INPUTS-1 loop
                    flattened_input(i) <= (others => '0');
                end loop;
            elsif current_state = FLATTEN then
                flat_index := 0;
                -- Flatten the 14x14 matrix to 144 elements (taking only 12x12 center)
                for i in 1 to 12 loop
                    for j in 1 to 12 loop
                        if matrice(i,j) = 9 then
                            flattened_input(flat_index) <= to_signed(127, 8);  -- White pixel
                        else
                            flattened_input(flat_index) <= to_signed(-128, 8); -- Black pixel
                        end if;
                        flat_index := flat_index + 1;
                    end loop;
                end loop;
            end if;
        end if;
    end process;
    
    -- FC1 Layer computation process
    main_compute_process: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                current_state <= IDLE;
                compute_index <= 0;
                input_index <= 0;
                accumulator <= (others => '0');
                done <= '0';
                predicted_digit <= (others => '0');
                confidence <= (others => '0');
                max_value <= (others => '0');
                max_index <= 0;
                search_index <= 0;
            else
                case current_state is
                    when IDLE =>
                        if start = '1' then
                            current_state <= FLATTEN;
                            done <= '0';
                        end if;
                        
                    when FLATTEN =>
                        current_state <= COMPUTE_FC1;
                        compute_index <= 0;
                        input_index <= 0;
                        accumulator <= (others => '0');
                        
                    when COMPUTE_FC1 =>
                        -- Multiply-accumulate operation for FC1
                        accumulator <= accumulator + (flattened_input(input_index) * FC1_WEIGHTS(compute_index, input_index));
                        
                        if input_index = FC1_INPUTS-1 then
                            -- Store the result and apply ReLU activation
                            fc1_results(compute_index) <= accumulator(15 downto 0);
                            
                            -- ReLU activation: max(0, x)
                            if accumulator(23) = '0' and accumulator /= 0 then
                                fc1_activated(compute_index) <= '1';
                            else
                                fc1_activated(compute_index) <= '0';
                            end if;
                            
                            if compute_index = FC1_OUTPUTS-1 then
                                current_state <= COMPUTE_FC2;
                                compute_index <= 0;
                                input_index <= 0;
                                accumulator <= (others => '0');
                            else
                                compute_index <= compute_index + 1;
                                input_index <= 0;
                                accumulator <= (others => '0');
                            end if;
                        else
                            input_index <= input_index + 1;
                        end if;
                        
                    when COMPUTE_FC2 =>
                        -- FC2 layer: FC1 outputs (128) -> 64 outputs
                        -- Use a simplified approach - select every other FC1 output
                        if compute_index < FC2_OUTPUTS then
                            -- Simple mapping: take every other FC1 output and apply weights
                            if fc1_activated(compute_index * 2) = '1' then
                                fc2_results(compute_index) <= fc1_results(compute_index * 2)(15 downto 1);
                                fc2_activated(compute_index) <= '1';
                            else
                                fc2_results(compute_index) <= (others => '0');
                                fc2_activated(compute_index) <= '0';
                            end if;
                            
                            if compute_index = FC2_OUTPUTS-1 then
                                current_state <= COMPUTE_FC3;
                                compute_index <= 0;
                                input_index <= 0;
                                accumulator <= (others => '0');
                            else
                                compute_index <= compute_index + 1;
                            end if;
                        end if;
                        
                    when COMPUTE_FC3 =>
                        -- FC3 layer: 64 inputs -> 10 outputs (final classification)
                        if input_index < FC3_INPUTS then
                            -- Multiply-accumulate for FC3
                            if fc2_activated(input_index) = '1' then
                                accumulator <= accumulator + (resize(fc2_results(input_index)(7 downto 0), 8) * FC3_WEIGHTS(compute_index, input_index));
                            end if;
                            
                            if input_index = FC3_INPUTS-1 then
                                -- Store FC3 result
                                fc3_results(compute_index) <= accumulator(15 downto 0);
                                
                                if compute_index = FC3_OUTPUTS-1 then
                                    current_state <= FIND_MAX;
                                    search_index <= 0;
                                    max_value <= fc3_results(0);
                                    max_index <= 0;
                                else
                                    compute_index <= compute_index + 1;
                                    input_index <= 0;
                                    accumulator <= (others => '0');
                                end if;
                            else
                                input_index <= input_index + 1;
                            end if;
                        end if;
                        
                    when FIND_MAX =>
                        -- Find the maximum value (predicted digit)
                        if search_index < FC3_OUTPUTS then
                            if fc3_results(search_index) > max_value then
                                max_value <= fc3_results(search_index);
                                max_index <= search_index;
                            end if;
                            
                            if search_index = FC3_OUTPUTS-1 then
                                current_state <= DONE_STATE;
                                predicted_digit <= std_logic_vector(to_unsigned(max_index, 4));
                                confidence <= std_logic_vector(max_value(7 downto 0));
                            else
                                search_index <= search_index + 1;
                            end if;
                        end if;
                        
                    when DONE_STATE =>
                        done <= '1';
                        if start = '0' then
                            current_state <= IDLE;
                        end if;
                        
                end case;
            end if;
        end if;
    end process;

end Behavioral;
