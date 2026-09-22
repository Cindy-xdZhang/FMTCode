void GenericLocalOptimization3d::computeWithOutFilter(Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives)
	{

		//for performance consideration and we always need construct discrete observer field u which demand discrete input field v, here we demand inputField be discrete!
		if (!inputField.HasDiscreteData())
		{
			inputField.ReSampleAnlyticalToDiscreteData();
		}

		using namespace Eigen;
		using Float = float;
		constexpr int taylorOrder = 1;//hiper paramter used for  EInvariance::Displacement
		const auto& invariance = mInvariance;
		auto info = inputField.getPrintOutGridInfo();
		LOG_D("comput observer with infor: %s", info.c_str());

		std::string timerString = "GGTCompute EInvariance::Objective";
		if (invariance == EInvariance::Displacement)
			timerString = "GGTCompute EInvariance::Displacement";

		Vector2i timesteps = ClampTimstep2ValidRange(inputField, timestepStart, timestepEnd);
		timestepStart = timesteps(0);
		timestepEnd = timesteps(1);

		// Get the information on the domain
		double minTime = inputField.GetMinTime();
		double maxTime = inputField.GetMaxTime();
		int numberOfTimeSteps = inputField.GetNumberOfTimeSteps();
		double timeSpacing = (maxTime - minTime) * (1.0 / double(numberOfTimeSteps - 1));

		const Vector3i res = inputField.GetSpatialGridSize();
		Eigen::Vector3d minDomainBounds = inputField.GetSpatialMin();
		Eigen::Vector3d maxDomainBounds = inputField.GetSpatialMax();
		Eigen::Vector3d domainSize = (maxDomainBounds - minDomainBounds);
		Eigen::Vector3d gridSpacing(domainSize(0) * (1.0 / double(res(0) - 1)), domainSize(1) * (1.0 / double(res(1) - 1)), domainSize(2) * (1.0 / double(res(2) - 1)));

		std::array<int, 4> dims = { res(0), res(1), res(2), numberOfTimeSteps };
		std::array<double, 4> spacing = { gridSpacing(0), gridSpacing(1), gridSpacing(2), timeSpacing };
		std::array<double, 4> boundsMin = { minDomainBounds(0), minDomainBounds(1), minDomainBounds(2), minTime };

		// need store observed time derivatives?
		bool storeObservedTimeDerivatives = observedTimeDerivatives.GetNumberOfTimeSteps() == inputField.GetNumberOfTimeSteps();

		// NEW: copy input data and compute derivatives

		std::vector<int> timeIndices(numberOfTimeSteps);
		std::iota(timeIndices.begin(), timeIndices.end(), 0);
        timeIndices = {1};

		// In 3D, M is a 3 ?12 matrix MTM*u=Mtv' ;  u is unknow shape 12x1; v' is dv/dt  shape is 3x1
		int systemSize = 12;
		if (invariance == EInvariance::Objective)
			systemSize = 12;
		else if (invariance == EInvariance::Displacement)
		{
			//EApproximation::Taylor:
			systemSize = taylorOrder * taylorOrder * taylorOrder + 6 * taylorOrder * taylorOrder + 11 * taylorOrder + 6;

		}

		using EigenMatrixX = Eigen::MatrixXd;
		using EigenVectorX = Eigen::VectorXd;

		Eigen::Matrix<float, Eigen::Dynamic, 3>& out_data_v_minus_u = resultVminusUfield.GetData();

		STARTTIMER_STRING(timerString);
		//		// Parallel loop over time slices
		std::for_each(policy, timeIndices.begin(), timeIndices.end(),
			[&](int tid) {
				int points_computed_this_slice = 0;
				std::string timerInfo = "GGT solving for timestep " + std::to_string(tid);

				STARTTIMER_STRING(timerInfo);
				const auto time = tid * timeSpacing + minTime;

				std::vector<EigenMatrixX> _MTM(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, systemSize));
				std::vector<EigenVectorX> _MTb(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, 1));
				std::vector<EigenVectorX> _b(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, 1));
				std::vector<bool> _HasM(res.x() * res.y() * res.z(), false);
				// chunk size, meaning each thread will process 16 iterations of the loop at a time before being assigned the next chunk.

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
				for (int z = 0; z < res.z(); ++z)
					for (int y = 0; y < res.y(); ++y)
						for (int x = 0; x < res.x(); ++x) {
							EigenMatrixX MTM(systemSize, systemSize);
							MTM.setZero();
							EigenVectorX MTb(systemSize, 1);
							MTb.setZero();

							_MTM[z * res.x() * res.y() + y * res.x() + x] = MTM;
							_MTb[z * res.x() * res.y() + y * res.x() + x] = MTb;
							_b[z * res.x() * res.y() + y * res.x() + x] = MTb;
						}

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
				for (int iz = 0; iz < res.z(); ++iz) {
					double z = boundsMin[2] + iz * spacing[2];
					for (int iy = 0; iy < res.y(); ++iy) {
						double y = boundsMin[1] + iy * spacing[1];
						for (int ix = 0; ix < res.x(); ++ix) {
							double x = boundsMin[0] + ix * spacing[0];
							int linIndex = iz * res.x() * res.y() + iy * res.x() + ix;
							{
								Eigen::Vector3f pos = Eigen::Vector3f(x, y, z);

								Eigen::Vector3f X = pos;
								Eigen::Vector3f V = inputField.GetVector(pos, (float)time);
								if (V.isZero())
									continue;
								points_computed_this_slice++;
								Eigen::Matrix3f XT = skewMat(X);
								Eigen::Matrix3f VT = skewMat(V);
								Eigen::Matrix3f J = inputField.getVelocityGradientTensor(ix, iy, iz, tid);
								Eigen::Matrix3f F = -J * XT + VT;
								Eigen::Vector3f Jxv = -J * X + V;

								// setup the matrix M (test matrix multiplication...)
								EigenMatrixX M(3, systemSize);
								if (invariance == EInvariance::Objective)
								{
									M(0, 0) = (Float)F(0, 0);
									M(0, 1) = (Float)F(0, 1);
									M(0, 2) = (Float)F(0, 2);
									M(0, 3) = (Float)J(0, 0);
									M(0, 4) = (Float)J(0, 1);
									M(0, 5) = (Float)J(0, 2);
									M(0, 6) = (Float)XT(0, 0);
									M(0, 7) = (Float)XT(0, 1);
									M(0, 8) = (Float)XT(0, 2);
									M(0, 9) = -1;
									M(0, 10) = 0;
									M(0, 11) = 0;
									M(1, 0) = (Float)F(1, 0);
									M(1, 1) = (Float)F(1, 1);
									M(1, 2) = (Float)F(1, 2);
									M(1, 3) = (Float)J(1, 0);
									M(1, 4) = (Float)J(1, 1);
									M(1, 5) = (Float)J(1, 2);
									M(1, 6) = (Float)XT(1, 0);
									M(1, 7) = (Float)XT(1, 1);
									M(1, 8) = (Float)XT(1, 2);
									M(1, 9) = 0;
									M(1, 10) = -1;
									M(1, 11) = 0;
									M(2, 0) = (Float)F(2, 0);
									M(2, 1) = (Float)F(2, 1);
									M(2, 2) = (Float)F(2, 2);
									M(2, 3) = (Float)J(2, 0);
									M(2, 4) = (Float)J(2, 1);
									M(2, 5) = (Float)J(2, 2);
									M(2, 6) = (Float)XT(2, 0);
									M(2, 7) = (Float)XT(2, 1);
									M(2, 8) = (Float)XT(2, 2);
									M(2, 9) = 0;
									M(2, 10) = 0;
									M(2, 11) = -1;
								}
								else if (invariance == EInvariance::Displacement)
								{
									int index = 0;
									for (int m = 0; m <= taylorOrder; ++m)
									{
										for (int i = m; i >= 0; --i)
										{
											for (int j = m - i; j >= 0; --j)
											{
												int k = m - i - j;

												Eigen::Matrix3f f, g;
												ComputeDisplacementSystemMatrixCoefficients(i, j, k, X, V, J, f, g);
												M(0, index * 6 + 0) = (Float)f(0, 0);		M(1, index * 6 + 0) = (Float)f(1, 0);		M(2, index * 6 + 0) = (Float)f(2, 0);
												M(0, index * 6 + 1) = (Float)f(0, 1);		M(1, index * 6 + 1) = (Float)f(1, 1);		M(2, index * 6 + 1) = (Float)f(2, 1);
												M(0, index * 6 + 2) = (Float)f(0, 2);		M(1, index * 6 + 2) = (Float)f(1, 2);		M(2, index * 6 + 2) = (Float)f(2, 2);

												M(0, index * 6 + 3) = (Float)g(0, 0);		M(1, index * 6 + 3) = (Float)g(1, 0);		M(2, index * 6 + 3) = (Float)g(2, 0);
												M(0, index * 6 + 4) = (Float)g(0, 1);		M(1, index * 6 + 4) = (Float)g(1, 1);		M(2, index * 6 + 4) = (Float)g(2, 1);
												M(0, index * 6 + 5) = (Float)g(0, 2);		M(1, index * 6 + 5) = (Float)g(1, 2);		M(2, index * 6 + 5) = (Float)g(2, 2);
												index += 1;
											}
										}
									}

								}


								auto MT = M.transpose();
								auto dt = inputField.getPartialDerivativeT(ix, iy, iz, tid);
								if (invariance == EInvariance::Displacement)
									dt *= -1;
								Eigen::Vector3d b(dt.x(), dt.y(), dt.z());

								_MTM[linIndex] = MT * M;
								_MTb[linIndex] = MT * b;
								_HasM[linIndex] = true;

								///	store b for computing observed time derivatives
								_b[linIndex] = b;
							}
						} // end for x
					} // end for y
				} // end for z
				std::string debugInfo = "GGT solving timestep " + std::to_string(tid) + ". points to compute this slice: " + std::to_string(points_computed_this_slice);
				LOG_D("%s", debugInfo.c_str());

				// -----------
				if (UseSummedAreaTables) {
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
					for (int z = 0; z < res.z(); ++z)
						for (int y = 0; y < res.y(); ++y)
							for (int x = 1; x < res.x(); ++x) {
								if (!_HasM[z * res.x() * res.y() + y * res.x() + (x - 1)])
									continue;
								_MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[z * res.x() * res.y() + y * res.x() + (x - 1)];
								_MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[z * res.x() * res.y() + y * res.x() + (x - 1)];
							}

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
					for (int z = 0; z < res.z(); ++z)
						for (int x = 0; x < res.x(); ++x)
							for (int y = 1; y < res.y(); ++y) {
								if (!_HasM[z * res.x() * res.y() + (y - 1) * res.x() + x])
									continue;
								_MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[z * res.x() * res.y() + (y - 1) * res.x() + x];
								_MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[z * res.x() * res.y() + (y - 1) * res.x() + x];
							}

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
					for (int x = 0; x < res.x(); ++x)
						for (int y = 0; y < res.y(); ++y)
							for (int z = 1; z < res.z(); ++z) {
								if (!_HasM[(z - 1) * res.x() * res.y() + y * res.x() + x])
									continue;
								_MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[(z - 1) * res.x() * res.y() + y * res.x() + x];
								_MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[(z - 1) * res.x() * res.y() + y * res.x() + x];
							}
				}
				// -----------

				//				//solving
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
				for (int z = 0; z < res.z(); ++z) {
					for (int y = 0; y < res.y(); ++y)
						for (int x = 0; x < res.x(); ++x) {
							if (!_HasM[z * res.x() * res.y() + y * res.x() + x])
								continue;

							EigenMatrixX MTM(systemSize, systemSize);
							MTM.setZero();
							EigenVectorX MTb(systemSize, 1);
							MTb.setZero();

							int x1 = std::min(std::max(0, x - NeighborhoodU), res.x() - 1);
							int y1 = std::min(std::max(0, y - NeighborhoodU), res.y() - 1);
							int z1 = std::min(std::max(0, z - NeighborhoodU), res.z() - 1);
							int x2 = std::min(std::max(0, x + NeighborhoodU), res.x() - 1);
							int y2 = std::min(std::max(0, y + NeighborhoodU), res.y() - 1);
							int z2 = std::min(std::max(0, z + NeighborhoodU), res.z() - 1);

							if (UseSummedAreaTables) {
								MTM = _MTM[z2 * res.x() * res.y() + y2 * res.x() + x2] - _MTM[z1 * res.x() * res.y() + y2 * res.x() + x2]
									- _MTM[z2 * res.x() * res.y() + y1 * res.x() + x2] - _MTM[z2 * res.x() * res.y() + y2 * res.x() + x1]
									+ _MTM[z2 * res.x() * res.y() + y1 * res.x() + x1] + _MTM[z1 * res.x() * res.y() + y2 * res.x() + x1]
									+ _MTM[z1 * res.x() * res.y() + y1 * res.x() + x2] - _MTM[z1 * res.x() * res.y() + y1 * res.x() + x1];

								MTb = _MTb[z2 * res.x() * res.y() + y2 * res.x() + x2] - _MTb[z1 * res.x() * res.y() + y2 * res.x() + x2]
									- _MTb[z2 * res.x() * res.y() + y1 * res.x() + x2] - _MTb[z2 * res.x() * res.y() + y2 * res.x() + x1]
									+ _MTb[z2 * res.x() * res.y() + y1 * res.x() + x1] + _MTb[z1 * res.x() * res.y() + y2 * res.x() + x1]
									+ _MTb[z1 * res.x() * res.y() + y1 * res.x() + x2] - _MTb[z1 * res.x() * res.y() + y1 * res.x() + x1];
							}
							else {
								for (int wz = z1; wz <= z2; ++wz)
									for (int wy = y1; wy <= y2; ++wy)
										for (int wx = x1; wx <= x2; ++wx) {
											MTM += _MTM[wz * res[0] * res[1] + wy * res[0] + wx];
											MTb += _MTb[wz * res[0] * res[1] + wy * res[0] + wx];
										}
							}

							Eigen::Vector3f pos = Eigen::Vector3f(boundsMin[0] + x * spacing[0], boundsMin[1] + y * spacing[1], boundsMin[2] + z * spacing[2]);
							Eigen::Vector3f X = pos;
							Eigen::Vector3f V = inputField.GetVector(pos, (float)time);

							// solve
							//const double epsilon = 1e-7; // Tiny scalar to stabilize the solve
							//MTM.diagonal().array() += epsilon;
							EigenVectorX uu = MTM.fullPivHouseholderQr().solve(MTb);
							double uu0 = uu(0);
							double uu1 = uu(1);
							double uu2 = uu(2);
							double uu3 = uu(3);
							double uu4 = uu(4);
							double uu5 = uu(5);
							Eigen::Vector3f U123(uu0, uu1, uu2);
							Eigen::Vector3f U456(uu3, uu4, uu5);
							// Eigen::Matrix3f U123T = skewMat(U123);
							Eigen::Matrix3f XT = skewMat(X);
							Eigen::Matrix3f VT = skewMat(V);

							Eigen::Matrix3f J;
							// compute v_new and J_new at the center
							Eigen::Vector3f vnew(0, 0, 0);
							Eigen::Matrix3f Jnew = Eigen::Matrix3f::Identity();




							if (invariance == EInvariance::Objective) [[likely]]
							{
								vnew = V + U123.cross(X) + U456;
							}
							else if (invariance == EInvariance::Similarity)
							{
								assert(false);
								vnew = V + U123.cross(X) - ((double)uu(12)) * X + U456;
							}
							else if (invariance == EInvariance::Affine)
							{
								assert(false);
								double h00_1 = -uu(0);
								double h10_1 = -uu(1);
								double h20_1 = -uu(2);
								double h01_1 = -uu(3);
								double h11_1 = -uu(4);
								double h21_1 = -uu(5);
								double h02_1 = -uu(6);
								double h12_1 = -uu(7);
								double h22_1 = -uu(8);
								Eigen::Matrix3f H1 = makeEigenMatrix3f(h00_1, h01_1, h02_1,
									h10_1, h11_1, h12_1,
									h20_1, h21_1, h22_1);
								Eigen::Vector3f k1(uu(9), uu(10), uu(11));

								vnew = V + H1 * X + k1;
							}
							else if (invariance == EInvariance::Displacement)
							{
								//if (approximation == EApproximation::Taylor)
								{
									// construct velocity
									{
										Eigen::Vector3f Ft(0, 0, 0);
										for (int m = 0; m <= taylorOrder; ++m)
										{
											for (int i = 0; i <= m; ++i)
											{
												for (int j = 0; j <= m - i; ++j)
												{
													int fu_i = i, fu_j = j, fu_k = m - i - j;
													// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
													int linear_fu = (fu_i + fu_j + fu_k) * (fu_i + fu_j + fu_k + 1) * (fu_i + fu_j + fu_k + 2) / 3 + (fu_j + fu_k) * (fu_j + fu_k + 1) + 2 * fu_k;
													double fu = uu(3 * linear_fu + 0);
													double fv = uu(3 * linear_fu + 1);
													double fw = uu(3 * linear_fu + 2);

													int fac_i = factorial(fu_i);
													int fac_j = factorial(fu_j);
													int fac_k = factorial(fu_k);
													Ft += pow(X.x(), fu_i) * pow(X.y(), fu_j) * pow(X.z(), fu_k) / (fac_i * fac_j * fac_k) * Eigen::Vector3f(fu, fv, fw);
												}
											}
										}
										vnew = V + Ft;
									}
									// construct Jacobian
									{
										Eigen::Matrix3f nablaFt = Eigen::Matrix3f::Zero();
										for (int m = 0; m <= taylorOrder; ++m)
										{
											for (int i = 0; i <= m; ++i)
											{
												for (int j = 0; j <= m - i - 1; ++j)
												{
													int fx_i = i + 1, fx_j = j, fx_k = m - i - j - 1;
													int fy_i = i, fy_j = j + 1, fy_k = m - i - j - 1;
													int fz_i = i, fz_j = j, fz_k = m - i - j;

													// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
													int linear_fx = (fx_i + fx_j + fx_k) * (fx_i + fx_j + fx_k + 1) * (fx_i + fx_j + fx_k + 2) / 3 + (fx_j + fx_k) * (fx_j + fx_k + 1) + 2 * fx_k;
													int linear_fy = (fy_i + fy_j + fy_k) * (fy_i + fy_j + fy_k + 1) * (fy_i + fy_j + fy_k + 2) / 3 + (fy_j + fy_k) * (fy_j + fy_k + 1) + 2 * fy_k;
													int linear_fz = (fz_i + fz_j + fz_k) * (fz_i + fz_j + fz_k + 1) * (fz_i + fz_j + fz_k + 2) / 3 + (fz_j + fz_k) * (fz_j + fz_k + 1) + 2 * fz_k;
													double fxu = uu(3 * linear_fx + 0);
													double fxv = uu(3 * linear_fx + 1);
													double fxw = uu(3 * linear_fx + 2);

													double fyu = uu(3 * linear_fy + 0);
													double fyv = uu(3 * linear_fy + 1);
													double fyw = uu(3 * linear_fy + 2);

													double fzu = uu(3 * linear_fz + 0);
													double fzv = uu(3 * linear_fz + 1);
													double fzw = uu(3 * linear_fz + 2);

													int fac_i = factorial(i);
													int fac_j = factorial(j);
													int fac_k = factorial(m - i - j - 1);
													nablaFt += pow(X.x(), i) * pow(X.y(), j) * pow(X.z(), m - i - j - 1) / (fac_i * fac_j * fac_k) *
														makeEigenMatrix3f(
															fxu, fyu, fzu,
															fxv, fyv, fzv,
															fxw, fyw, fzw);
												}
											}
										}
										Jnew = J + nablaFt;
									}
								}
							}



							//output->SetVertexDataAt({ x, y, z }, vnew);
							unsigned long idx3D = z * dims[0] * dims[1] + y * dims[0] + x;
							unsigned long tupleIdx = (unsigned long)(tid * dims[0] * dims[1] * dims[2]) + idx3D;

							out_data_v_minus_u.row(tupleIdx) = vnew;
							if (invariance == EInvariance::Objective && storeObservedTimeDerivatives) {
								EigenVectorX sum_b(3, 1);
								sum_b.setZero();
								for (int wz = z1; wz <= z2; ++wz)
									for (int wy = y1; wy <= y2; ++wy)
										for (int wx = x1; wx <= x2; ++wx) {
											sum_b += _b[wz * res[0] * res[1] + wy * res[0] + wx];
										}

								// norm of |Mx-b|^2= xTMTMx-2bTMx +btb
								Eigen::VectorXd Mx_b_2 = uu.transpose() * MTM * uu - 2 * MTb.transpose() * uu + sum_b.transpose() * sum_b;
								double observedTimeDerivative = Mx_b_2.norm();
								observedTimeDerivatives.SetValue(x, y, z, tid, observedTimeDerivative);
							}

						}//iterate x

				}//iterate z

				STOPTIMER_STRING(timerInfo);
			});

		STOPTIMER_STRING(timerString);


		// compute (v-u)*-1.0 + v
		Eigen::Matrix<float, Eigen::Dynamic, 3>& out_actual_u = resultUfield.GetData();
		const auto& actual_v = inputField.GetDataView();
		out_actual_u = out_data_v_minus_u * -1.0 + actual_v;
		/*	for (int it = 0; it < dims[2]; it++) {
		int timeSliceOffset = it * dims[0] * dims[1] * 2;
		for (int iy = 0; iy < dims[1]; iy++) {
				int pixelRowOffset = timeSliceOffset + iy * dims[0] * 2;
				for (int ix = 0; ix < dims[0]; ix++) {
						int tupleIdx = it * dims[0] * dims[1] + iy * dims[0] + ix;
						int pixelOffset = pixelRowOffset + ix * 2;
						out_actual_u(tupleIdx, 0) += vectorField[it](iy, ix)(0);
						out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
						out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
				}
		}
	}*/

		return;
	}
