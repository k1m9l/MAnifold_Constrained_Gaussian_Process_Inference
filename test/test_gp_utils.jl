# test/test_gp_utils.jl

using Test
using MagiJl.GaussianProcess # Access mat2band (assuming it's exported or qualified)
using BandedMatrices
using LinearAlgebra

@testset "GP Utilities" begin

    @testset "mat2band" begin
        # Test case 1: Simple 4x4 matrix
        dense_mat = Float64[
            1 2 0 0;
            3 4 5 0;
            0 6 7 8;
            0 0 9 10
        ]
        l, u = 1, 1 # Bandwidths

        # Explicitly qualify the function call
        banded_mat = MagiJl.GaussianProcess.mat2band(dense_mat, l, u)

        @test banded_mat isa BandedMatrix
        @test BandedMatrices.bandwidths(banded_mat) == (l, u)
        @test size(banded_mat) == size(dense_mat)

        # Check elements within the band
        @test banded_mat[1, 1] == 1.0
        @test banded_mat[1, 2] == 2.0
        @test banded_mat[2, 1] == 3.0
        @test banded_mat[2, 2] == 4.0
        @test banded_mat[2, 3] == 5.0
        @test banded_mat[3, 2] == 6.0
        @test banded_mat[3, 3] == 7.0
        @test banded_mat[3, 4] == 8.0
        @test banded_mat[4, 3] == 9.0
        @test banded_mat[4, 4] == 10.0

        # Check elements outside the band are zero
        @test banded_mat[1, 3] == 0.0
        @test banded_mat[1, 4] == 0.0
        @test banded_mat[2, 4] == 0.0
        @test banded_mat[3, 1] == 0.0
        @test banded_mat[4, 1] == 0.0
        @test banded_mat[4, 2] == 0.0

        # Test case 2: Wider bands
        dense_mat2 = reshape(collect(1.0:16.0), 4, 4)
        l2, u2 = 2, 1
        banded_mat2 = MagiJl.GaussianProcess.mat2band(dense_mat2, l2, u2)
        @test BandedMatrices.bandwidths(banded_mat2) == (l2, u2)
        @test banded_mat2[3, 1] == dense_mat2[3, 1] # Should be included (l=2)
        @test banded_mat2[4, 1] == 0.0             # Outside lower band
        @test banded_mat2[1, 2] == dense_mat2[1, 2] # Should be included (u=1)
        @test banded_mat2[1, 3] == 0.0             # Outside upper band

        # Test case 3: Bandwidth >= size - 1 (should be same as dense)
        l3, u3 = 3, 3
        banded_mat3 = MagiJl.GaussianProcess.mat2band(dense_mat2, l3, u3)
        @test BandedMatrices.bandwidths(banded_mat3) == (l3, u3)
        @test banded_mat3 == dense_mat2 # All elements should be copied

    end

    @testset "mat2band with Various Bandwidths" begin
        test_mat = reshape(1.0:25.0, 5, 5)
        
        # Test with bandwidth = 0 (only diagonal)
        band0 = MagiJl.GaussianProcess.mat2band(test_mat, 0, 0)
        @test band0[1,1] == 1.0
        @test band0[3,3] == 13.0
        @test band0[1,2] == 0.0  # Off-diagonal should be zero
        
        # Test with full bandwidth (should preserve original matrix)
        band_full = MagiJl.GaussianProcess.mat2band(test_mat, 4, 4)
        @test band_full == test_mat
        
        # Test with asymmetric bandwidths
        band_asym = MagiJl.GaussianProcess.mat2band(test_mat, 2, 1)
        @test band_asym[3,2] == test_mat[3,2]  # Within lower band
        @test band_asym[2,3] == test_mat[2,3]  # Within upper band
        @test band_asym[4,1] == 0.0  # Outside lower band
        @test band_asym[1,4] == 0.0  # Outside upper band
    end

end