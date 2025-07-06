-- Quantized Neural Network Inference Entity
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.quantized_weights.all;

entity quantized_nn_inference is
    Port ( 
        clk : in STD_LOGIC;
        rst : in STD_LOGIC;
        start : in STD_LOGIC;
        input_data : in signed(7 downto 0);  -- 8-bit signed input
        input_valid : in STD_LOGIC;
        output_class : out integer range 0 to 9;
        output_valid : out STD_LOGIC
    );
end quantized_nn_inference;

architecture Behavioral of quantized_nn_inference is
    
    -- Internal signals for layer computations
    type fc1_input_array is array (0 to FC1_INPUTS-1) of signed(7 downto 0);
    type fc1_output_array is array (0 to FC1_OUTPUTS-1) of signed(15 downto 0);
    type fc2_output_array is array (0 to FC2_OUTPUTS-1) of signed(15 downto 0);
    
    signal fc1_inputs : fc1_input_array;
    signal fc1_outputs : fc1_output_array;
    signal fc2_outputs : fc2_output_array;
    
    signal input_counter : integer range 0 to FC1_INPUTS-1;
    signal computation_state : integer range 0 to 3;
    
begin

    process(clk, rst)
    begin
        if rst = '1' then
            input_counter <= 0;
            computation_state <= 0;
            output_valid <= '0';
            
        elsif rising_edge(clk) then
            case computation_state is
                when 0 => -- Input loading state
                    if input_valid = '1' then
                        fc1_inputs(input_counter) <= input_data;
                        if input_counter = FC1_INPUTS-1 then
                            input_counter <= 0;
                            computation_state <= 1;
                        else
                            input_counter <= input_counter + 1;
                        end if;
                    end if;
                    
                when 1 => -- FC1 computation
                    -- Implement matrix multiplication for FC1 layer
                    -- This would require multiple clock cycles for full computation
                    computation_state <= 2;
                    
                when 2 => -- FC2 computation
                    -- Implement matrix multiplication for FC2 layer
                    computation_state <= 3;
                    
                when 3 => -- Output generation
                    -- Find maximum output and generate class prediction
                    output_valid <= '1';
                    computation_state <= 0;
                    
                when others =>
                    computation_state <= 0;
            end case;
        end if;
    end process;

end Behavioral;