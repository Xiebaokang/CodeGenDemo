; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @aligned_alloc(i32, i32)

define void @Matmul_m1024n1024k1024_randomString(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
  %4 = call i32 @llvm.amdgcn.workgroup.id.x(), !range !2
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !range !2
  %6 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i32))
  %7 = addrspacecast ptr %6 to ptr addrspace(3)
  %8 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i32))
  %9 = addrspacecast ptr %8 to ptr addrspace(3)
  %10 = call i32 @llvm.amdgcn.workitem.id.x(), !range !3
  %11 = call i32 @llvm.amdgcn.workitem.id.y(), !range !3
  %12 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i32))
  %13 = addrspacecast ptr %12 to ptr addrspace(5)
  %14 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i32))
  %15 = addrspacecast ptr %14 to ptr addrspace(5)
  %16 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i32))
  %17 = addrspacecast ptr %16 to ptr addrspace(5)
  %18 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i32))
  %19 = addrspacecast ptr %18 to ptr addrspace(5)
  %20 = call ptr @aligned_alloc(i32 16, i32 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i32))
  %21 = addrspacecast ptr %20 to ptr addrspace(5)
  br label %22

22:                                               ; preds = %34, %3
  %23 = phi i32 [ %35, %34 ], [ 0, %3 ]
  %24 = icmp slt i32 %23, 8
  br i1 %24, label %25, label %36

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %29, %25
  %27 = phi i32 [ %33, %29 ], [ 0, %25 ]
  %28 = icmp slt i32 %27, 8
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = mul i32 %23, 8
  %31 = add i32 %30, %27
  %32 = getelementptr float, ptr addrspace(5) %21, i32 %31
  store float 0.000000e+00, ptr addrspace(5) %32, align 4
  %33 = add i32 %27, 1
  br label %26

34:                                               ; preds = %26
  %35 = add i32 %23, 1
  br label %22

36:                                               ; preds = %22
  %37 = mul i32 %10, 8
  %38 = icmp slt i32 %11, 0
  %39 = sub i32 -1, %11
  %40 = select i1 %38, i32 %39, i32 %11
  %41 = sdiv i32 %40, 2
  %42 = sub i32 -1, %41
  %43 = select i1 %38, i32 %42, i32 %41
  %44 = add i32 %37, %43
  %45 = mul i32 %4, 128
  %46 = add i32 %44, %45
  %47 = srem i32 %11, 2
  %48 = icmp slt i32 %47, 0
  %49 = add i32 %47, 2
  %50 = select i1 %48, i32 %49, i32 %47
  %51 = mul i32 %50, 4
  %52 = mul i32 %46, 1024
  %53 = add i32 %52, %51
  %54 = getelementptr float, ptr addrspace(1) %0, i32 %53
  %55 = load <4 x float>, ptr addrspace(1) %54, align 4
  store <4 x float> %55, ptr addrspace(5) %13, align 4
  %56 = mul i32 %10, 16
  %57 = add i32 %56, %11
  %58 = icmp slt i32 %57, 0
  %59 = sub i32 -1, %57
  %60 = select i1 %58, i32 %59, i32 %57
  %61 = sdiv i32 %60, 32
  %62 = sub i32 -1, %61
  %63 = select i1 %58, i32 %62, i32 %61
  %64 = srem i32 %57, 32
  %65 = icmp slt i32 %64, 0
  %66 = add i32 %64, 32
  %67 = select i1 %65, i32 %66, i32 %64
  %68 = mul i32 %67, 4
  %69 = mul i32 %5, 128
  %70 = add i32 %68, %69
  %71 = mul i32 %63, 1024
  %72 = add i32 %71, %70
  %73 = getelementptr float, ptr addrspace(1) %1, i32 %72
  %74 = load <4 x float>, ptr addrspace(1) %73, align 4
  store <4 x float> %74, ptr addrspace(5) %15, align 4
  br label %75

75:                                               ; preds = %78, %36
  %76 = phi i32 [ %86, %78 ], [ 0, %36 ]
  %77 = icmp slt i32 %76, 4
  br i1 %77, label %78, label %87

78:                                               ; preds = %75
  %79 = getelementptr float, ptr addrspace(5) %13, i32 %76
  %80 = load <1 x float>, ptr addrspace(5) %79, align 4
  %81 = add i32 %51, %76
  %82 = mul i32 %81, 128
  %83 = add i32 0, %82
  %84 = add i32 %83, %44
  %85 = getelementptr float, ptr addrspace(3) %7, i32 %84
  store <1 x float> %80, ptr addrspace(3) %85, align 4
  %86 = add i32 %76, 1
  br label %75

87:                                               ; preds = %75
  %88 = load <4 x float>, ptr addrspace(5) %15, align 4
  %89 = mul i32 %63, 128
  %90 = add i32 0, %89
  %91 = add i32 %90, %68
  %92 = getelementptr float, ptr addrspace(3) %9, i32 %91
  store <4 x float> %88, ptr addrspace(3) %92, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %93 = icmp slt i32 %67, 0
  %94 = sub i32 -1, %67
  %95 = select i1 %93, i32 %94, i32 %67
  %96 = sdiv i32 %95, 4
  %97 = sub i32 -1, %96
  %98 = select i1 %93, i32 %97, i32 %96
  %99 = icmp slt i32 %63, 0
  %100 = sub i32 -1, %63
  %101 = select i1 %99, i32 %100, i32 %63
  %102 = sdiv i32 %101, 4
  %103 = sub i32 -1, %102
  %104 = select i1 %99, i32 %103, i32 %102
  %105 = mul i32 %104, 8
  %106 = add i32 %98, %105
  %107 = mul i32 %106, 4
  %108 = add i32 0, %107
  %109 = getelementptr float, ptr addrspace(3) %7, i32 %108
  %110 = load <4 x float>, ptr addrspace(3) %109, align 4
  %111 = getelementptr float, ptr addrspace(5) %17, i32 0
  store <4 x float> %110, ptr addrspace(5) %111, align 4
  %112 = add i32 %104, 2
  %113 = mul i32 %112, 8
  %114 = add i32 %98, %113
  %115 = mul i32 %114, 4
  %116 = add i32 0, %115
  %117 = getelementptr float, ptr addrspace(3) %7, i32 %116
  %118 = load <4 x float>, ptr addrspace(3) %117, align 4
  %119 = getelementptr float, ptr addrspace(5) %17, i32 4
  store <4 x float> %118, ptr addrspace(5) %119, align 4
  %120 = srem i32 %11, 4
  %121 = icmp slt i32 %120, 0
  %122 = add i32 %120, 4
  %123 = select i1 %121, i32 %122, i32 %120
  %124 = srem i32 %63, 4
  %125 = icmp slt i32 %124, 0
  %126 = add i32 %124, 4
  %127 = select i1 %125, i32 %126, i32 %124
  %128 = mul i32 %127, 4
  %129 = add i32 %123, %128
  %130 = mul i32 %129, 4
  %131 = add i32 0, %130
  %132 = getelementptr float, ptr addrspace(3) %9, i32 %131
  %133 = load <4 x float>, ptr addrspace(3) %132, align 4
  %134 = getelementptr float, ptr addrspace(5) %19, i32 0
  store <4 x float> %133, ptr addrspace(5) %134, align 4
  %135 = add i32 %127, 4
  %136 = mul i32 %135, 4
  %137 = add i32 %123, %136
  %138 = mul i32 %137, 4
  %139 = add i32 0, %138
  %140 = getelementptr float, ptr addrspace(3) %9, i32 %139
  %141 = load <4 x float>, ptr addrspace(3) %140, align 4
  %142 = getelementptr float, ptr addrspace(5) %19, i32 4
  store <4 x float> %141, ptr addrspace(5) %142, align 4
  br label %143

143:                                              ; preds = %298, %87
  %144 = phi i32 [ %324, %298 ], [ 0, %87 ]
  %145 = icmp slt i32 %144, 1024
  br i1 %145, label %146, label %325

146:                                              ; preds = %143
  %147 = sub i32 1008, %144
  %148 = icmp sge i32 %147, 0
  br i1 %148, label %149, label %160

149:                                              ; preds = %146
  %150 = add i32 %144, 8
  %151 = add i32 %51, %150
  %152 = add i32 %52, %151
  %153 = getelementptr float, ptr addrspace(1) %0, i32 %152
  %154 = load <4 x float>, ptr addrspace(1) %153, align 4
  store <4 x float> %154, ptr addrspace(5) %13, align 4
  %155 = add i32 %63, %150
  %156 = mul i32 %155, 1024
  %157 = add i32 %156, %70
  %158 = getelementptr float, ptr addrspace(1) %1, i32 %157
  %159 = load <4 x float>, ptr addrspace(1) %158, align 4
  store <4 x float> %159, ptr addrspace(5) %15, align 4
  br label %160

160:                                              ; preds = %203, %149, %146
  %161 = phi i32 [ %176, %203 ], [ 0, %149 ], [ 0, %146 ]
  br label %162

162:                                              ; preds = %160
  %163 = phi i32 [ %161, %160 ]
  %164 = icmp slt i32 %163, 7
  br i1 %164, label %165, label %231

165:                                              ; preds = %162
  %166 = icmp slt i32 %144, 0
  %167 = sub i32 -1, %144
  %168 = select i1 %166, i32 %167, i32 %144
  %169 = sdiv i32 %168, 8
  %170 = sub i32 -1, %169
  %171 = select i1 %166, i32 %170, i32 %169
  %172 = srem i32 %171, 2
  %173 = icmp slt i32 %172, 0
  %174 = add i32 %172, 2
  %175 = select i1 %173, i32 %174, i32 %172
  %176 = add i32 %163, 1
  %177 = mul i32 %175, 1024
  %178 = mul i32 %176, 128
  %179 = add i32 %177, %178
  %180 = add i32 %179, %107
  %181 = getelementptr float, ptr addrspace(3) %7, i32 %180
  %182 = load <4 x float>, ptr addrspace(3) %181, align 4
  %183 = srem i32 %176, 2
  %184 = icmp slt i32 %183, 0
  %185 = add i32 %183, 2
  %186 = select i1 %184, i32 %185, i32 %183
  %187 = mul i32 %186, 8
  %188 = add i32 %187, 0
  %189 = getelementptr float, ptr addrspace(5) %17, i32 %188
  store <4 x float> %182, ptr addrspace(5) %189, align 4
  %190 = add i32 %179, %115
  %191 = getelementptr float, ptr addrspace(3) %7, i32 %190
  %192 = load <4 x float>, ptr addrspace(3) %191, align 4
  %193 = add i32 %187, 4
  %194 = getelementptr float, ptr addrspace(5) %17, i32 %193
  store <4 x float> %192, ptr addrspace(5) %194, align 4
  %195 = add i32 %179, %130
  %196 = getelementptr float, ptr addrspace(3) %9, i32 %195
  %197 = load <4 x float>, ptr addrspace(3) %196, align 4
  %198 = getelementptr float, ptr addrspace(5) %19, i32 %188
  store <4 x float> %197, ptr addrspace(5) %198, align 4
  %199 = add i32 %179, %138
  %200 = getelementptr float, ptr addrspace(3) %9, i32 %199
  %201 = load <4 x float>, ptr addrspace(3) %200, align 4
  %202 = getelementptr float, ptr addrspace(5) %19, i32 %193
  store <4 x float> %201, ptr addrspace(5) %202, align 4
  br label %203

203:                                              ; preds = %229, %165
  %204 = phi i32 [ %230, %229 ], [ 0, %165 ]
  %205 = icmp slt i32 %204, 8
  br i1 %205, label %206, label %160

206:                                              ; preds = %203
  br label %207

207:                                              ; preds = %210, %206
  %208 = phi i32 [ %228, %210 ], [ 0, %206 ]
  %209 = icmp slt i32 %208, 8
  br i1 %209, label %210, label %229

210:                                              ; preds = %207
  %211 = mul i32 %204, 8
  %212 = add i32 %211, %208
  %213 = getelementptr float, ptr addrspace(5) %21, i32 %212
  %214 = load float, ptr addrspace(5) %213, align 4
  %215 = srem i32 %163, 2
  %216 = icmp slt i32 %215, 0
  %217 = add i32 %215, 2
  %218 = select i1 %216, i32 %217, i32 %215
  %219 = mul i32 %218, 8
  %220 = add i32 %219, %204
  %221 = getelementptr float, ptr addrspace(5) %17, i32 %220
  %222 = load float, ptr addrspace(5) %221, align 4
  %223 = add i32 %219, %208
  %224 = getelementptr float, ptr addrspace(5) %19, i32 %223
  %225 = load float, ptr addrspace(5) %224, align 4
  %226 = fmul float %222, %225
  %227 = fadd float %226, %214
  store float %227, ptr addrspace(5) %213, align 4
  %228 = add i32 %208, 1
  br label %207

229:                                              ; preds = %207
  %230 = add i32 %204, 1
  br label %203

231:                                              ; preds = %162
  br i1 %148, label %232, label %274

232:                                              ; preds = %231
  br label %233

233:                                              ; preds = %236, %232
  %234 = phi i32 [ %256, %236 ], [ 0, %232 ]
  %235 = icmp slt i32 %234, 4
  br i1 %235, label %236, label %257

236:                                              ; preds = %233
  %237 = getelementptr float, ptr addrspace(5) %13, i32 %234
  %238 = load <1 x float>, ptr addrspace(5) %237, align 4
  %239 = icmp slt i32 %144, 0
  %240 = sub i32 -1, %144
  %241 = select i1 %239, i32 %240, i32 %144
  %242 = sdiv i32 %241, 8
  %243 = sub i32 -1, %242
  %244 = select i1 %239, i32 %243, i32 %242
  %245 = add i32 %244, 1
  %246 = srem i32 %245, 2
  %247 = icmp slt i32 %246, 0
  %248 = add i32 %246, 2
  %249 = select i1 %247, i32 %248, i32 %246
  %250 = add i32 %51, %234
  %251 = mul i32 %249, 1024
  %252 = mul i32 %250, 128
  %253 = add i32 %251, %252
  %254 = add i32 %253, %44
  %255 = getelementptr float, ptr addrspace(3) %7, i32 %254
  store <1 x float> %238, ptr addrspace(3) %255, align 4
  %256 = add i32 %234, 1
  br label %233

257:                                              ; preds = %233
  %258 = load <4 x float>, ptr addrspace(5) %15, align 4
  %259 = icmp slt i32 %144, 0
  %260 = sub i32 -1, %144
  %261 = select i1 %259, i32 %260, i32 %144
  %262 = sdiv i32 %261, 8
  %263 = sub i32 -1, %262
  %264 = select i1 %259, i32 %263, i32 %262
  %265 = add i32 %264, 1
  %266 = srem i32 %265, 2
  %267 = icmp slt i32 %266, 0
  %268 = add i32 %266, 2
  %269 = select i1 %267, i32 %268, i32 %266
  %270 = mul i32 %269, 1024
  %271 = add i32 %270, %89
  %272 = add i32 %271, %68
  %273 = getelementptr float, ptr addrspace(3) %9, i32 %272
  store <4 x float> %258, ptr addrspace(3) %273, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  br label %274

274:                                              ; preds = %257, %231
  br label %275

275:                                              ; preds = %296, %274
  %276 = phi i32 [ %297, %296 ], [ 0, %274 ]
  %277 = icmp slt i32 %276, 8
  br i1 %277, label %278, label %298

278:                                              ; preds = %275
  br label %279

279:                                              ; preds = %282, %278
  %280 = phi i32 [ %295, %282 ], [ 0, %278 ]
  %281 = icmp slt i32 %280, 8
  br i1 %281, label %282, label %296

282:                                              ; preds = %279
  %283 = mul i32 %276, 8
  %284 = add i32 %283, %280
  %285 = getelementptr float, ptr addrspace(5) %21, i32 %284
  %286 = load float, ptr addrspace(5) %285, align 4
  %287 = add i32 8, %276
  %288 = getelementptr float, ptr addrspace(5) %17, i32 %287
  %289 = load float, ptr addrspace(5) %288, align 4
  %290 = add i32 8, %280
  %291 = getelementptr float, ptr addrspace(5) %19, i32 %290
  %292 = load float, ptr addrspace(5) %291, align 4
  %293 = fmul float %289, %292
  %294 = fadd float %293, %286
  store float %294, ptr addrspace(5) %285, align 4
  %295 = add i32 %280, 1
  br label %279

296:                                              ; preds = %279
  %297 = add i32 %276, 1
  br label %275

298:                                              ; preds = %275
  %299 = icmp slt i32 %144, 0
  %300 = sub i32 -1, %144
  %301 = select i1 %299, i32 %300, i32 %144
  %302 = sdiv i32 %301, 8
  %303 = sub i32 -1, %302
  %304 = select i1 %299, i32 %303, i32 %302
  %305 = add i32 %304, 1
  %306 = srem i32 %305, 2
  %307 = icmp slt i32 %306, 0
  %308 = add i32 %306, 2
  %309 = select i1 %307, i32 %308, i32 %306
  %310 = mul i32 %309, 1024
  %311 = add i32 %310, 0
  %312 = add i32 %311, %130
  %313 = getelementptr float, ptr addrspace(3) %9, i32 %312
  %314 = load <4 x float>, ptr addrspace(3) %313, align 4
  store <4 x float> %314, ptr addrspace(5) %134, align 4
  %315 = add i32 %311, %138
  %316 = getelementptr float, ptr addrspace(3) %9, i32 %315
  %317 = load <4 x float>, ptr addrspace(3) %316, align 4
  store <4 x float> %317, ptr addrspace(5) %142, align 4
  %318 = add i32 %311, %107
  %319 = getelementptr float, ptr addrspace(3) %7, i32 %318
  %320 = load <4 x float>, ptr addrspace(3) %319, align 4
  store <4 x float> %320, ptr addrspace(5) %111, align 4
  %321 = add i32 %311, %115
  %322 = getelementptr float, ptr addrspace(3) %7, i32 %321
  %323 = load <4 x float>, ptr addrspace(3) %322, align 4
  store <4 x float> %323, ptr addrspace(5) %119, align 4
  %324 = add i32 %144, 8
  br label %143

325:                                              ; preds = %143
  br label %326

326:                                              ; preds = %329, %325
  %327 = phi i32 [ %340, %329 ], [ 0, %325 ]
  %328 = icmp slt i32 %327, 4
  br i1 %328, label %329, label %341

329:                                              ; preds = %326
  %330 = mul i32 %327, 8
  %331 = add i32 %330, 0
  %332 = getelementptr float, ptr addrspace(5) %21, i32 %331
  %333 = load <4 x float>, ptr addrspace(5) %332, align 4
  %334 = add i32 %45, %107
  %335 = add i32 %334, %327
  %336 = add i32 %69, %130
  %337 = mul i32 %335, 1024
  %338 = add i32 %337, %336
  %339 = getelementptr float, ptr addrspace(1) %2, i32 %338
  store <4 x float> %333, ptr addrspace(1) %339, align 4
  %340 = add i32 %327, 1
  br label %326

341:                                              ; preds = %326
  br label %342

342:                                              ; preds = %345, %341
  %343 = phi i32 [ %356, %345 ], [ 0, %341 ]
  %344 = icmp slt i32 %343, 4
  br i1 %344, label %345, label %357

345:                                              ; preds = %342
  %346 = mul i32 %343, 8
  %347 = add i32 %346, 4
  %348 = getelementptr float, ptr addrspace(5) %21, i32 %347
  %349 = load <4 x float>, ptr addrspace(5) %348, align 4
  %350 = add i32 %45, %107
  %351 = add i32 %350, %343
  %352 = add i32 %69, %138
  %353 = mul i32 %351, 1024
  %354 = add i32 %353, %352
  %355 = getelementptr float, ptr addrspace(1) %2, i32 %354
  store <4 x float> %349, ptr addrspace(1) %355, align 4
  %356 = add i32 %343, 1
  br label %342

357:                                              ; preds = %342
  br label %358

358:                                              ; preds = %361, %357
  %359 = phi i32 [ %373, %361 ], [ 0, %357 ]
  %360 = icmp slt i32 %359, 4
  br i1 %360, label %361, label %374

361:                                              ; preds = %358
  %362 = add i32 %359, 4
  %363 = mul i32 %362, 8
  %364 = add i32 %363, 0
  %365 = getelementptr float, ptr addrspace(5) %21, i32 %364
  %366 = load <4 x float>, ptr addrspace(5) %365, align 4
  %367 = add i32 %45, %115
  %368 = add i32 %367, %359
  %369 = add i32 %69, %130
  %370 = mul i32 %368, 1024
  %371 = add i32 %370, %369
  %372 = getelementptr float, ptr addrspace(1) %2, i32 %371
  store <4 x float> %366, ptr addrspace(1) %372, align 4
  %373 = add i32 %359, 1
  br label %358

374:                                              ; preds = %358
  br label %375

375:                                              ; preds = %378, %374
  %376 = phi i32 [ %390, %378 ], [ 0, %374 ]
  %377 = icmp slt i32 %376, 4
  br i1 %377, label %378, label %391

378:                                              ; preds = %375
  %379 = add i32 %376, 4
  %380 = mul i32 %379, 8
  %381 = add i32 %380, 4
  %382 = getelementptr float, ptr addrspace(5) %21, i32 %381
  %383 = load <4 x float>, ptr addrspace(5) %382, align 4
  %384 = add i32 %45, %115
  %385 = add i32 %384, %376
  %386 = add i32 %69, %138
  %387 = mul i32 %385, 1024
  %388 = add i32 %387, %386
  %389 = getelementptr float, ptr addrspace(1) %2, i32 %388
  store <4 x float> %383, ptr addrspace(1) %389, align 4
  %390 = add i32 %376, 1
  br label %375

391:                                              ; preds = %375
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workgroup.id.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.y() #0

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nofree nounwind willreturn }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @Matmul_m1024n1024k1024_randomString, !"kernel", i32 1}
!2 = !{i32 0, i32 8}
!3 = !{i32 0, i32 16}
