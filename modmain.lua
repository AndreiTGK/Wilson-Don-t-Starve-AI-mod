-- Wilson AI Observer
-- modmain.lua
--
-- Exports Wilson's local perception every tick to a persistent string file.
-- Designed for Don't Starve (base) and Don't Starve Together.

local GLOBAL         = GLOBAL
local TheSim         = GLOBAL.TheSim
local pairs          = GLOBAL.pairs
local ipairs         = GLOBAL.ipairs
local table          = GLOBAL.table
local string         = GLOBAL.string
local math           = GLOBAL.math
local tostring       = GLOBAL.tostring
local tonumber       = GLOBAL.tonumber
local type           = GLOBAL.type
local print          = GLOBAL.print
local pcall          = GLOBAL.pcall
local io             = GLOBAL.io
local ACTIONS        = GLOBAL.ACTIONS
local BufferedAction = GLOBAL.BufferedAction
local Vector3        = GLOBAL.Vector3
local GROUND         = GLOBAL.GROUND

local TheWorld  = nil
local thePlayer = nil

-- ─── Config ─────────────────────────────────────────────────────────────────

local PERCEPTION_RADIUS = 40
local MEMORY_RADIUS     = 60
local EXPORT_PERIOD     = 0.1

local JUNK_TAGS = {"FX", "NOCLICK", "DECOR", "INLIMBO", "player"}

-- ─── Speed control ──────────────────────────────────────────────────────────

local current_speed = 1

local function apply_speed(n)
    n = math.max(1, math.min(20, math.floor(n)))
    if n == current_speed then return end
    current_speed = n
    TheSim:SetTimeScale(n)
    print("[WilsonAI] Speed: " .. n .. "x")
end

local function read_speed_file()
    local f = io.open(MODROOT .. "wilson_ai_speed", "r")
    if f then
        local data = f:read("*all")
        f:close()
        local n = tonumber(data and data:match("%d+"))
        if n then apply_speed(n) end
    end
end

-- ─── JSON helpers ───────────────────────────────────────────────────────────

local function safe_str(v)
    if v == nil then return "null" end
    if type(v) == "boolean" then return v and "true" or "false" end
    if type(v) == "number" then
        if v ~= v then return "0" end
        return tostring(math.floor(v * 100) / 100)
    end
    return '"' .. tostring(v):gsub('"', '\\"') .. '"'
end

local function vec_to_json(t)
    return "[" .. table.concat(t, ",") .. "]"
end

-- ─── Sound pulses ───────────────────────────────────────────────────────────

local sound_pulses = {
    hound_near=0, hound_attack=0, low_hp=0, low_sanity=0, low_hunger=0,
    dusk=0, dawn=0, rain=0, thunder=0, boss_near=0, darkness=0, damage_taken=0,
}

local function pulse(event)
    if sound_pulses[event] ~= nil then sound_pulses[event] = 1 end
end

local function decay_pulses()
    for k in pairs(sound_pulses) do sound_pulses[k] = 0 end
end

local function get_pulses_json()
    local parts = {}
    for k, v in pairs(sound_pulses) do
        table.insert(parts, string.format('"%s":%d', k, v))
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

local item_cooldowns = {}

-- ─── Death memory ───────────────────────────────────────────────────────────

local death_memory = {}
local death_count  = 0
local MAX_DEATH_MEMORIES = 50

local function record_death(player)
    death_count = death_count + 1
    local x, _, z = player.Transform:GetWorldPosition()
    local hp  = player.components.health and player.components.health.currenthealth or 0
    local hun = player.components.hunger and player.components.hunger.current       or 0
    local san = player.components.sanity and player.components.sanity.current       or 0
    local clock_d = GLOBAL.GetClock and GLOBAL.GetClock()
    local day = (clock_d and clock_d.GetNumCycles and clock_d:GetNumCycles())
             or (TheWorld and TheWorld.state and TheWorld.state.cycles)
             or 0

    local mem = string.format(
        '{"death_id":%d,"x":%s,"z":%s,"hp":%s,"hunger":%s,"sanity":%s,"day":%s}',
        death_count, safe_str(x), safe_str(z),
        safe_str(hp), safe_str(hun), safe_str(san), safe_str(day)
    )

    if #death_memory >= MAX_DEATH_MEMORIES then table.remove(death_memory, 1) end
    table.insert(death_memory, mem)
end

local function get_death_memory_json()
    return "[" .. table.concat(death_memory, ",") .. "]"
end

-- ─── Entity perception ──────────────────────────────────────────────────────

local INTERESTING = {
    berries=true, carrot=true, mushroom=true, pumpkin=true, watermelon=true,
    fish=true, meat=true, cookedmeat=true, morsels=true, cookedmorsels=true,
    tree=true, evergreen=true, deciduoustree=true, palmtree=true,
    rocks=true, flint=true, gold=true, log=true, pinecone=true,
    grass=true, sapling=true, reeds=true, flowers=true,
    rabbit=true, bird=true, pig=true, beefalo=true, spider=true,
    hound=true, tentacle=true, shadow=true, deerclops=true,
    fire=true, firepit=true, campfire=true, chest=true,
    sciencemachine=true, alchemyengine=true, crockpot=true, icebox=true,
    researchlab=true,
}

local function serialize_entity(ent, px, pz, radius_label)
    if not ent or not ent:IsValid() or not ent.Transform then return nil end
    local ex, _, ez = ent.Transform:GetWorldPosition()
    local dx, dz = ex - px, ez - pz
    local dist  = math.sqrt(dx*dx + dz*dz)
    local angle = math.floor(math.atan2(dz, dx) * 180 / math.pi)
    local hp_pct = "null"
    if ent.components and ent.components.health then
        local h = ent.components.health
        if h.maxhealth and h.maxhealth > 0 then
            hp_pct = safe_str(h.currenthealth / h.maxhealth)
        end
    end
    return string.format('{"prefab":%s,"dist":%s,"angle":%s,"hp_pct":%s,"radius":%s}',
        safe_str(ent.prefab), safe_str(dist), safe_str(angle), hp_pct, safe_str(radius_label))
end

local function sort_by_dist(ents, px, pz)
    table.sort(ents, function(a, b)
        if not a.Transform then return false end
        if not b.Transform then return true end
        local ax, _, az = a.Transform:GetWorldPosition()
        local bx, _, bz = b.Transform:GetWorldPosition()
        return (ax-px)^2 + (az-pz)^2 < (bx-px)^2 + (bz-pz)^2
    end)
end

-- ─── Inventory serialization ─────────────────────────────────────────────────

local function item_dur_pct(item)
    if item.components.finiteuses then
        local ok, v = pcall(function() return item.components.finiteuses:GetPercent() end)
        if ok and type(v) == "number" then return safe_str(v) end
    end
    return "null"
end

local function item_fresh_pct(item)
    if item.components.perishable then
        local ok, v = pcall(function() return item.components.perishable:GetPercent() end)
        if ok and type(v) == "number" then return safe_str(v) end
    end
    return "null"
end

local function serialize_inventory(player)
    local inv = player.components and player.components.inventory
    if not inv then return "[]" end

    local slots = {}
    for i = 1, inv:GetNumSlots() do
        local item = inv:GetItemInSlot(i)
        if item then
            local count = 1
            if item.components.stackable then
                count = item.components.stackable.stacksize or 1
            end
            table.insert(slots, string.format('{"p":%s,"n":%d,"d":%s,"f":%s}',
                safe_str(item.prefab), count, item_dur_pct(item), item_fresh_pct(item)))
        else
            table.insert(slots, '{"p":"","n":0,"d":null,"f":null}')
        end
    end

    if inv.equipslots then
        for _, item in pairs(inv.equipslots) do
            if item then
                table.insert(slots, string.format('{"p":%s,"n":1,"d":%s,"f":null,"e":1}',
                    safe_str(item.prefab), item_dur_pct(item)))
            else
                table.insert(slots, '{"p":"","n":0,"d":null,"f":null}')
            end
        end
    end
    return vec_to_json(slots)
end

-- ─── Stat-driven sound pulses ───────────────────────────────────────────────

local function check_stat_pulses(player)
    local h = player.components.health
    local n = player.components.hunger
    local s = player.components.sanity

    if h and h.maxhealth  and h.maxhealth  > 0 and (h.currenthealth / h.maxhealth)  < 0.25 then pulse("low_hp")     end
    if n and n.max        and n.max        > 0 and (n.current       / n.max)        < 0.2  then pulse("low_hunger") end
    if s and s.maxsanity  and s.maxsanity  > 0 and (s.current       / s.maxsanity)  < 0.25 then pulse("low_sanity") end

    if TheWorld and TheWorld.state and TheWorld.state.phase == "night" then
        local x, y, z = player.Transform:GetWorldPosition()
        local lights = TheSim:FindEntities(x, y, z, 8, {"_light"})
        if not lights or #lights == 0 then pulse("darkness") end
    end
end

-- ─── Crafting system ─────────────────────────────────────────────────────────

local CRAFT_SLOT_NAMES = {
    [0]="axe",       [1]="pickaxe",        [2]="torch",  [3]="rope",
    [4]="campfire",  [5]="sciencemachine",  [6]="spear",  [7]="icebox",
}
local N_CRAFT_SLOTS = 8
local PLACE_DIST    = 2.0
local pending_placement = nil

local PLACEABLE_RECIPES = { campfire=true, sciencemachine=true, icebox=true }

local function serialize_craft_slots(player)
    local parts   = {}
    local builder = player.components.builder
    for i = 0, N_CRAFT_SLOTS - 1 do
        local name      = CRAFT_SLOT_NAMES[i]
        local is_place  = PLACEABLE_RECIPES[name] and 1 or 0
        local can_build = 0
        if builder then
            local ok, result = pcall(function() return builder:CanBuild(name) end)
            can_build = (ok and result) and 1 or 0
        end
        table.insert(parts, string.format(
            '{"available":1,"product":%s,"ingredients_met":%d,"is_placeable":%d}',
            safe_str(name), can_build, is_place))
    end
    return "[" .. table.concat(parts, ",") .. "]"
end

local function execute_craft(player, slot_idx)
    local name = CRAFT_SLOT_NAMES[slot_idx]
    if not name or pending_placement then return end
    local builder = player.components.builder
    if not builder then return end
    local ok, can = pcall(function() return builder:CanBuild(name) end)
    if not (ok and can) then return end
    if PLACEABLE_RECIPES[name] then
        pending_placement = name
        print("[WilsonAI] Holding for placement: " .. name)
    else
        local px, py, pz = player.Transform:GetWorldPosition()
        pcall(function() builder:DoBuild(name, Vector3(px, py, pz), nil) end)
    end
end

local function execute_place(player, dir_idx)
    if not pending_placement then return end
    local builder = player.components.builder
    if not builder then return end
    local px, py, pz = player.Transform:GetWorldPosition()
    local pt
    if dir_idx == 8 then
        pt = Vector3(px, py, pz)
    else
        local angle_rad = MOVE_ANGLES[dir_idx % 8] * math.pi / 180
        pt = Vector3(px + math.cos(angle_rad) * PLACE_DIST, py, pz - math.sin(angle_rad) * PLACE_DIST)
    end
    local ok = pcall(function() builder:DoBuild(pending_placement, pt, nil) end)
    if ok then
        print("[WilsonAI] Placed: " .. pending_placement)
        pending_placement = nil
    end
end

-- ─── Action execution ────────────────────────────────────────────────────────

local RAD2DEG = 180 / math.pi
local MOVE_ANGLES = {}
local _DIRS = {
    [0]={ 0, 1}, [1]={ 1, 1}, [2]={ 1, 0}, [3]={ 1,-1},
    [4]={ 0,-1}, [5]={-1,-1}, [6]={-1, 0}, [7]={-1, 1},
}
for k, v in pairs(_DIRS) do MOVE_ANGLES[k] = math.atan2(-v[2], v[1]) * RAD2DEG end

local current_action = nil
local ACTION_FILE    = MODROOT .. "wilson_ai_action.txt"

local function read_action_file()
    local f = io.open(ACTION_FILE, "r")
    if f then
        local data = f:read("*all")
        f:close()
        local n = tonumber(data and data:match("%-?%d+"))
        if n then current_action = n end
    end
end

local function get_target_in_dir(player, angle, radius)
    local px, py, pz = player.Transform:GetWorldPosition()
    local ents    = TheSim:FindEntities(px, py, pz, radius)
    local best_ent = nil
    local min_diff = 45

    for _, ent in ipairs(ents) do
        if ent ~= player and ent.entity:IsVisible() then
            local ex, _, ez = ent.Transform:GetWorldPosition()
            local ent_angle = math.atan2(-(ez - pz), (ex - px)) * RAD2DEG
            local diff = math.abs((ent_angle - angle + 180) % 360 - 180)
            if diff < min_diff then
                min_diff  = diff
                best_ent  = ent
            end
        end
    end
    return best_ent
end

local function execute_current_action(player)
    if current_action == nil or not player or not player:IsValid() then return end

    local a      = current_action
    local loco   = player.components.locomotor
    local combat = player.components.combat
    local inv    = player.components.inventory

    if a >= 0 and a <= 7 then
        if loco then loco:RunInDirection(MOVE_ANGLES[a]) end

    elseif a == 8 then
        if loco then loco:Stop() end

    elseif a >= 10 and a <= 17 then
        if player.sg and player.sg:HasStateTag("busy") then return end
        local target = get_target_in_dir(player, MOVE_ANGLES[a - 10], 3)
        if target then
            local act_picker = player.components.playeractionpicker
            if act_picker then
                local acts  = nil
                local t_pos = target:GetPosition()
                if act_picker.GetLeftClickActions then
                    acts = act_picker:GetLeftClickActions(t_pos, target)
                end
                if not acts or #acts == 0 then
                    acts = act_picker:GetSceneActions(target)
                end
                if acts and #acts > 0 then
                    player.components.locomotor:PushAction(acts[1], true)
                end
            end
        end

    elseif a >= 20 and a <= 27 then
        if player.sg and player.sg:HasStateTag("busy") then return end
        local target = get_target_in_dir(player, MOVE_ANGLES[a - 20], 4)
        if target and combat and combat:CanTarget(target) and loco then
            loco:PushAction(BufferedAction(player, target, ACTIONS.ATTACK), true)
        end

    elseif a >= 30 and a <= 39 then
        if player.sg and player.sg:HasStateTag("busy") then return end
        local slot = (a - 30) + 1
        if inv then
            local item = inv:GetItemInSlot(slot)
            if item then
                local current_time = GLOBAL.GetTime()
                local last_used    = item_cooldowns[item] or 0
                if current_time - last_used > 1.0 then
                    if item.components.edible then
                        player.components.locomotor:PushAction(
                            GLOBAL.BufferedAction(player, item, GLOBAL.ACTIONS.EAT), true)
                        item_cooldowns[item] = current_time
                    elseif item.components.equippable and not item.components.equippable:IsEquipped() then
                        player.components.locomotor:PushAction(
                            GLOBAL.BufferedAction(player, item, GLOBAL.ACTIONS.EQUIP), true)
                        item_cooldowns[item] = current_time
                    end
                end
            end
        end

    elseif a >= 40 and a <= 47 then execute_craft(player, a - 40)
    elseif a >= 48 and a <= 56 then execute_place(player, a - 48)
    end
end

-- ─── Tile sensing ───────────────────────────────────────────────────────────

local TILE_STEP = 4

local function sample_tile_hazard(x, z)
    if not TheWorld or not TheWorld.Map then return 0 end
    local tile = TheWorld.Map:GetTileAtPoint(x, 0, z)
    if tile == GROUND.IMPASSABLE or tile == GROUND.INVALID then return 1 end
    local ok, wet = pcall(function() return TheWorld.Map:IsWater(tile) end)
    return (ok and wet) and 1 or 0
end

local function get_tiles_json(px, pz)
    local t = {
        sample_tile_hazard(px,             pz),
        sample_tile_hazard(px,             pz + TILE_STEP),
        sample_tile_hazard(px + TILE_STEP, pz),
        sample_tile_hazard(px,             pz - TILE_STEP),
        sample_tile_hazard(px - TILE_STEP, pz),
    }
    return "[" .. table.concat(t, ",") .. "]"
end

-- ─── World border ───────────────────────────────────────────────────────────

local world_half_w     = 512
local world_half_h     = 512
local world_size_cached = false

local function try_cache_world_size()
    if world_size_cached or not TheWorld or not TheWorld.Map then return end
    local ok, w, h = pcall(function() return TheWorld.Map:GetSize() end)
    if ok and w and h then
        world_half_w     = w * 2
        world_half_h     = h * 2
        world_size_cached = true
        print(string.format("[WilsonAI] World size: %dx%d tiles", w, h))
    end
end

local function get_border_json(px, pz)
    try_cache_world_size()
    local wx = math.max(0, math.min(1, (px + world_half_w) / (2 * world_half_w)))
    local wz = math.max(0, math.min(1, (pz + world_half_h) / (2 * world_half_h)))
    return string.format("[%s,%s]", safe_str(wx), safe_str(wz))
end

-- ─── Main export ────────────────────────────────────────────────────────────

local tick_counter = 0

local function export_state(player)
    if not player or not player:IsValid() or not player.Transform then return end
    tick_counter = tick_counter + 1

    read_speed_file()
    check_stat_pulses(player)

    local px, py, pz = player.Transform:GetWorldPosition()
    local health   = player.components.health
    local hunger   = player.components.hunger
    local sanity   = player.components.sanity
    local moisture = player.components.moisture

    local hp      = health   and health.currenthealth or 0
    local hp_max  = health   and health.maxhealth     or 150
    local hun     = hunger   and hunger.current       or 0
    local hun_max = hunger   and hunger.max           or 150
    local san     = sanity   and sanity.current       or 0
    local san_max = sanity   and sanity.maxsanity     or 200
    local wet     = moisture and moisture.moisture    or 0

    -- Temperature: 35 is comfortable; below 0 = freezing, above 70 = overheating
    local temperature = 35
    local temp_comp   = player.components.temperature
    if temp_comp and type(temp_comp.current) == "number" then
        temperature = temp_comp.current
    end

    local ws     = TheWorld and TheWorld.state or {}
    local clock  = GLOBAL.GetClock and GLOBAL.GetClock()
    local day    = (clock and clock.GetNumCycles and clock:GetNumCycles()) or ws.cycles or 0
    local phase  = (clock and clock.GetPhase     and clock:GetPhase())    or ws.phase  or "day"
    local season = (clock and clock.GetSeason    and clock:GetSeason())   or ws.season or "autumn"
    local raining = ws.israining or false

    -- Fraction through the current day/dusk/night cycle (0-1).
    -- clock.time is a DS clock field; fall back to 0.5 if unavailable.
    local phase_pct = 0.5
    if clock and type(clock.time) == "number" then
        phase_pct = clock.time
    end

    local near = TheSim:FindEntities(px, py, pz, PERCEPTION_RADIUS, nil, JUNK_TAGS) or {}
    local mem  = TheSim:FindEntities(px, py, pz, MEMORY_RADIUS,     nil, JUNK_TAGS) or {}
    sort_by_dist(near, px, pz)
    sort_by_dist(mem,  px, pz)

    local perceived, remembered, seen = {}, {}, {}

    for _, ent in ipairs(near) do
        if ent ~= player and INTERESTING[ent.prefab] then
            local s = serialize_entity(ent, px, pz, "near")
            if s then table.insert(perceived, s); seen[ent] = true end
        end
    end

    for _, ent in ipairs(mem) do
        if ent ~= player and not seen[ent] and INTERESTING[ent.prefab] then
            local s = serialize_entity(ent, px, pz, "memory")
            if s then table.insert(remembered, s) end
        end
    end

    local pending_json = pending_placement
        and string.format('{"active":1,"product":%s}', safe_str(pending_placement))
        or  '{"active":0,"product":null}'

    local json = string.format([[{
"self":{"hp":%s,"hp_max":%s,"hunger":%s,"hunger_max":%s,"sanity":%s,"sanity_max":%s,"wetness":%s,"temperature":%s,"x":%s,"z":%s},
"world":{"day":%s,"phase":%s,"phase_pct":%s,"season":%s,"is_raining":%s},
"inventory":%s,
"perceived":%s,
"remembered":%s,
"sounds":%s,
"speed":%d,
"death_memory":%s,
"death_count":%d,
"tick":%d,
"craft_slots":%s,
"pending_placement":%s,
"tiles":%s,
"border":%s
}]],
        safe_str(hp), safe_str(hp_max), safe_str(hun), safe_str(hun_max),
        safe_str(san), safe_str(san_max), safe_str(wet), safe_str(temperature),
        safe_str(px), safe_str(pz),
        safe_str(day), safe_str(phase), safe_str(phase_pct), safe_str(season), safe_str(raining),
        serialize_inventory(player), vec_to_json(perceived), vec_to_json(remembered),
        get_pulses_json(), current_speed, get_death_memory_json(), death_count, tick_counter,
        serialize_craft_slots(player), pending_json,
        get_tiles_json(px, pz), get_border_json(px, pz)
    )

    local f = io.open(MODROOT .. "wilson_ai_state", "w")
    if f then f:write(json .. "\n--END--"); f:close() end
    decay_pulses()
end

-- ─── Player attachment ──────────────────────────────────────────────────────

local function attach_to_player(player)
    if player._wilson_attached then return end
    player._wilson_attached = true
    thePlayer = player

    if TheWorld == nil then
        local ok, w = pcall(function() return GLOBAL.TheWorld end)
        if ok and w then TheWorld = w elseif GLOBAL.GetWorld then TheWorld = GLOBAL.GetWorld() end
    end

    player:ListenForEvent("death", function()
        record_death(player)
        local x, _, z = player.Transform:GetWorldPosition()
        local df = io.open(MODROOT .. "wilson_ai_death", "w")
        if df then
            df:write(string.format('{"event":"DEATH","death_count":%d,"x":%s,"z":%s}',
                death_count, safe_str(x), safe_str(z)))
            df:close()
        end

        player:DoTaskInTime(2.0, function()
            local current_slot = GLOBAL.SaveGameIndex:GetCurrentSaveSlot()
            local character    = player.prefab or "wilson"
            GLOBAL.SaveGameIndex:DeleteSlot(current_slot, function()
                GLOBAL.SaveGameIndex:StartSurvivalMode(current_slot, character, nil, function()
                    GLOBAL.StartNextInstance({
                        reset_action = GLOBAL.RESET_ACTION.LOAD_SLOT,
                        save_slot    = current_slot,
                    })
                end)
            end)
        end)
    end)

    -- Pulse damage_taken for any hit; also pulse hound_attack when a hound variant hits.
    player:ListenForEvent("attacked", function(inst, data)
        pulse("damage_taken")
        if data and data.attacker and data.attacker.prefab then
            if string.find(data.attacker.prefab, "hound") then
                pulse("hound_attack")
            end
        end
    end)

    if TheWorld then
        TheWorld:ListenForEvent("phasechanged", function(_, data)
            if data and data.newphase == "dusk" then pulse("dusk")
            elseif data and data.newphase == "day" then pulse("dawn") end
        end)
        TheWorld:ListenForEvent("rainstart",      function() pulse("rain")       end)
        TheWorld:ListenForEvent("lightningflash", function() pulse("thunder")    end)
        TheWorld:ListenForEvent("houndattack",    function() pulse("hound_near") end)
    end

    player:DoPeriodicTask(EXPORT_PERIOD, function() export_state(player)      end)
    player:DoPeriodicTask(0.1,           function() read_action_file(); execute_current_action(player) end)
end

-- ─── Hooks ───────────────────────────────────────────────────────────────────

AddSimPostInit(function() print("[WilsonAI] Sim loaded.") end)
AddPlayerPostInit(function(player) attach_to_player(player) end)

AddPlayerPostInit(function(player)
    player:DoTaskInTime(0.5, function()
        if player.sg and player.sg:HasStateTag("wakeup") then player.sg:GoToState("idle") end
        local x, y, z = player.Transform:GetWorldPosition()
        local ents = GLOBAL.TheSim:FindEntities(x, y, z, 30)
        for _, ent in ipairs(ents) do if ent.prefab == "maxwellintro" then ent:Remove() end end
        if player.HUD then player.HUD:Show() end
        if player.components.playercontroller then player.components.playercontroller:Enable(true) end
    end)
end)

AddGamePostInit(function()
    TheSim:SetTimeScale(1)
    current_speed = 1
    local player = GLOBAL.GetPlayer and GLOBAL.GetPlayer()
    if player then attach_to_player(player) end
end)
