# Unified game asset snapshots

This directory contains the bundled Stage 1 runtime closures for the six registered
`legacy_cli` games: Minecraft, Elden Ring, Elden Ring Nightreign, Apex Legends,
GTA5, and Cyberpunk 2077. These files originated as source snapshots, but the
public package applies portability edits such as removing environment-specific
endpoint defaults. The table below records the historical source-snapshot sizes
and hashes; it does not claim that the public files remain byte-for-byte identical.

`public_manifest.json` is the integrity manifest for the modified public-package
assets. It records the current byte length and SHA-256 digest of every bundled
`.py` and `.json` asset except the manifest itself. Use that file—not the
historical table below—to verify installed public assets.

| Game | Local asset | Upstream source | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| Apex Legends | `apex_legends/1a_Apex_Legends_segment_from_actions.py` | `Apex_Legends_scripts_parrellel/1a_Apex_Legends_segment_from_actions.py` | 40,991 | `5e0e71f6c297cc46c33e5ece73dbe5f945d872aecc59a2ab3aa43c006139fe8d` |
| Apex Legends | `apex_legends/1b_Apex_Legends_refine_with_gemini.py` | `Apex_Legends_scripts_parrellel/1b_Apex_Legends_refine_with_gemini.py` | 56,325 | `0c0b06b15a11736757b29a4e6ceaaf0c9419cdff2d103bd5c55a3b506c6a4823` |
| Apex Legends | `apex_legends/Apex_Legends_keybinds.json` | `Apex_Legends_scripts_parrellel/Apex_Legends_keybinds.json` | 854 | `71b49419785a7fb5598ad0c1b49192cb076de2aca7b8e0e2e19e62e700225a38` |
| Apex Legends | `apex_legends/Apex_Legends_taxonomy.py` | `Apex_Legends_scripts_parrellel/Apex_Legends_taxonomy.py` | 26,518 | `ef384252d088e36eb956e5b7cd4e3f080536029696e63b980206225655f7ba0c` |
| Apex Legends | `apex_legends/Apex_Legends_video_io.py` | `Apex_Legends_scripts_parrellel/Apex_Legends_video_io.py` | 11,111 | `8c84694b76b69ef4bc8e47c7537cb0783d5cf481eda8dd5fe64218d67fa0e14f` |
| Cyberpunk 2077 | `cyberpunk_2077/1a_Cyberpunk_2077_scripts_segment_from_actions.py` | `Cyberpunk_2077_scripts/1a_Cyberpunk_2077_scripts_segment_from_actions.py` | 34,260 | `bd4b4340109014bf599b8184bd9a80e160ab9aaf58244d6e8355be5a8dea0dbc` |
| Cyberpunk 2077 | `cyberpunk_2077/1b_Cyberpunk_2077_refine_with_gemini.py` | `Cyberpunk_2077_scripts/1b_Cyberpunk_2077_refine_with_gemini.py` | 43,629 | `1c55fef5e05be23926fa912dc892b3f038bfeab98f4bea595a502e8149dae6ab` |
| Cyberpunk 2077 | `cyberpunk_2077/Cyberpunk_2077_keybinds.json` | `Cyberpunk_2077_scripts/Cyberpunk_2077_keybinds.json` | 1,659 | `57fdee0ff706ac425120b2f857a60ba68d4b03249d72a5b502d40edef96b9d54` |
| Cyberpunk 2077 | `cyberpunk_2077/Cyberpunk_2077_taxonomy.py` | `Cyberpunk_2077_scripts/Cyberpunk_2077_taxonomy.py` | 32,563 | `eb96464e531a330ace3dfa3b7ffdfd6ec663dbbd2c27ed330018016e48a6a485` |
| Cyberpunk 2077 | `cyberpunk_2077/Cyberpunk_2077_video_io.py` | `Apex_Legends_scripts_parrellel/Apex_Legends_video_io.py` (shared helper) | 11,111 | `8c84694b76b69ef4bc8e47c7537cb0783d5cf481eda8dd5fe64218d67fa0e14f` |
| Elden Ring | `eldenring/1a_EldenRing_segment_from_actions.py` | `EldenRing_scripts/1a_EldenRing_segment_from_actions.py` | 22,050 | `48e7566bcbc783ffd2ddbc2ae092002e781d5bf42199e51f258ed2c9c8f4c1f8` |
| Elden Ring | `eldenring/1b_EldenRing_refine_with_gemini.py` | `EldenRing_scripts/1b_EldenRing_refine_with_gemini.py` | 26,996 | `4dd7f2707b2b5ac65e56d9d0f4f78c2900e40c2051100704c8a73c8b30ab47ea` |
| Elden Ring | `eldenring/EldenRing_keybinds.json` | `EldenRing_scripts/EldenRing_keybinds.json` | 2,477 | `9a14a62225ce374adf0836ac10ccabb1d5e06e048fd241bcf2a1ad1d5f942ed4` |
| Elden Ring | `eldenring/EldenRing_taxonomy.py` | `EldenRing_scripts/EldenRing_taxonomy.py` | 44,096 | `0869e56ece5380a7ae68b9c014cb89003eb99d55f950cd91074f9c0fa1a5a706` |
| Elden Ring | `eldenring/EldenRing_video_io.py` | `EldenRing_scripts/EldenRing_video_io.py` | 8,300 | `b2995b38ab9c9c2c579bd189fba642baaad5f35925d9c7d84a74028b8d41fb5b` |
| Elden Ring Nightreign | `eldenring_nightreign/1a_EldenRing-NightReign_segment_from_actions.py` | `EldenRing-NightReign_scripts/1a_EldenRing-NightReign_segment_from_actions.py` | 43,251 | `d1d4b1eda559274003ccc814fd4ee0a747e4fc36f7fa254eef0bf08bb7164b76` |
| Elden Ring Nightreign | `eldenring_nightreign/1b_EldenRing-NightReign_refine_with_gemini.py` | `EldenRing-NightReign_scripts/1b_EldenRing-NightReign_refine_with_gemini.py` | 55,532 | `f9f93e5256f8e90b0ce766ca075643a9635f54722f8b74b7613ccf8748bb317c` |
| Elden Ring Nightreign | `eldenring_nightreign/EldenRing_NightReign_keybinds.json` | `EldenRing-NightReign_scripts/EldenRing_NightReign_keybinds.json` | 2,225 | `2e2f380f3cc899c3a0db02c27fe5e1774caa363edd716ee660608b0aabc0188d` |
| Elden Ring Nightreign | `eldenring_nightreign/EldenRing_NightReign_taxonomy.py` | `EldenRing-NightReign_scripts/EldenRing_NightReign_taxonomy.py` | 29,388 | `c8b2bacc0815597933e82513c273ee9dab8eca7f4721d334aa9d7feace83f8b6` |
| Elden Ring Nightreign | `eldenring_nightreign/EldenRing_NightReign_video_io.py` | `EldenRing-NightReign_scripts/EldenRing_NightReign_video_io.py` | 11,420 | `6c4bdf78c0878f88f5bdd485cde70a96919bed937a7d715df0484403c103247c` |
| GTA5 | `gta5/1a_GTA5_scripts_segment_from_actions.py` | `GTA5_scripts/1a_GTA5_scripts_segment_from_actions.py` | 34,287 | `0c4ed7e235ba8e2ad14b86f63a35a4c9310b61254924bd326c89672ba29a6634` |
| GTA5 | `gta5/1b_GTA5_refine_with_gemini.py` | `GTA5_scripts/1b_GTA5_refine_with_gemini.py` | 47,809 | `d54d5a779ac66fe050c43607f801b5e655ac2715e2fa624b69ca0c5a2af92adf` |
| GTA5 | `gta5/GTA5_keybinds.json` | `GTA5_scripts/GTA5_keybinds.json` | 1,602 | `e570a3e7e12c3e19fbbc8ee8c954e169aa5386a15e768214c773f184e13419d0` |
| GTA5 | `gta5/GTA5_taxonomy.py` | `GTA5_scripts/GTA5_taxonomy.py` | 28,150 | `5dcd0cfc335839afef05d3f7db3a4d04ef00b71f7740be327fec9752ef7f0844` |
| Minecraft | `minecraft/1a_Minecraft_scripts_segment_from_actions.py` | `Minecraft_scripts_parrellel/1a_Minecraft_scripts_segment_from_actions.py` | 29,861 | `677d041d05fa1ac595cec4bfa9d94025289959b5298a32bd213eee292af31514` |
| Minecraft | `minecraft/1b_Minecraft_refine_with_gemini.py` | `Minecraft_scripts_parrellel/1b_Minecraft_refine_with_gemini.py` | 40,488 | `7d112f224d9bee4b0f1e8b48885e09dbc1c317f4018d927c5f989944dc647865` |
| Minecraft | `minecraft/Minecraft_keybinds.json` | `Minecraft_scripts_parrellel/Minecraft_keybinds.json` | 899 | `fd00aed850351106a8b1baa8c17e81853bda79a8a6f6a9c716697a9e507273ed` |
| Minecraft | `minecraft/Minecraft_taxonomy.py` | `Minecraft_scripts_parrellel/Minecraft_taxonomy.py` | 26,021 | `1e29ff6b0ce42e2bb04f4b0fa938982cda44ac943689ac4258b8d958aaba999b` |
| Minecraft | `minecraft/Minecraft_video_io.py` | `Minecraft_scripts_parrellel/Minecraft_video_io.py` | 9,756 | `bde548e6326d1116d89992c01728eef1442c92783fd091b4376cfc9246ee5151` |

Only the files named above are snapshots. All six `legacy_cli` profiles load
their Stage 1 scripts, taxonomy, keybinds, and helper modules exclusively from
this unified package tree.
