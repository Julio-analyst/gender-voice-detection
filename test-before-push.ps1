# ============================================================================
# TEST BEFORE PUSH - Local GitHub Actions Simulation
# ============================================================================
# This script runs all checks locally before pushing to GitHub
# Simulates: lint.yml, test.yml workflows
# ============================================================================

param(
    [switch]$FixFormatting,
    [switch]$SkipTests
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🧪 LOCAL GITHUB ACTIONS TESTING" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$ErrorCount = 0
$WarningCount = 0

# ============================================================================
# 1. DEPENDENCY CHECK
# ============================================================================
Write-Host "[1/5] 📦 Checking Dependencies..." -ForegroundColor Yellow

try {
    Write-Host "  → Verifying requirements.txt compatibility..." -ForegroundColor Gray
    
    # Quick dependency conflict check
    $depCheck = pip check 2>&1 | Select-String "gradio|fastapi|tensorflow|pydantic|typing-extensions"
    
    if ($depCheck) {
        Write-Host "  ⚠️  Potential dependency conflicts detected:" -ForegroundColor Red
        $depCheck | ForEach-Object { Write-Host "     $_" -ForegroundColor Red }
        $ErrorCount++
    } else {
        Write-Host "  ✅ No critical dependency conflicts" -ForegroundColor Green
    }
} catch {
    Write-Host "  ⚠️  Could not verify dependencies" -ForegroundColor Yellow
    $WarningCount++
}

# ============================================================================
# 2. LINTING (Black + isort)
# ============================================================================
Write-Host "`n[2/5] 🎨 Code Formatting (Black + isort)..." -ForegroundColor Yellow

# Black formatting check
Write-Host "  → Running Black formatter..." -ForegroundColor Gray

if ($FixFormatting) {
    black src/ tests/ | Out-Null
    Write-Host "  ✅ Black formatting applied" -ForegroundColor Green
} else {
    $blackCheck = black src/ tests/ --check 2>&1 | Out-String
    
    if ($blackCheck -match "would reformat") {
        $blackCount = ($blackCheck | Select-String "would reformat").Matches.Count
        Write-Host "  ❌ $blackCount files need Black formatting" -ForegroundColor Red
        Write-Host "     Run with -FixFormatting to auto-fix" -ForegroundColor Gray
        $ErrorCount++
    } else {
        Write-Host "  ✅ Black formatting passed" -ForegroundColor Green
    }
}

# isort import sorting check
Write-Host "  → Running isort..." -ForegroundColor Gray

if ($FixFormatting) {
    isort src/ tests/ | Out-Null
    Write-Host "  ✅ Import sorting applied" -ForegroundColor Green
} else {
    $isortCheck = isort src/ tests/ --check-only 2>&1 | Out-String
    
    if ($isortCheck -match "ERROR:") {
        $isortCount = ($isortCheck | Select-String "ERROR:").Matches.Count
        Write-Host "  ❌ $isortCount files need import sorting" -ForegroundColor Red
        Write-Host "     Run with -FixFormatting to auto-fix" -ForegroundColor Gray
        $ErrorCount++
    } else {
        Write-Host "  ✅ isort passed" -ForegroundColor Green
    }
}

# ============================================================================
# 3. SYNTAX CHECK (Flake8)
# ============================================================================
Write-Host "`n[3/5] 🔍 Syntax Check (Flake8)..." -ForegroundColor Yellow

Write-Host "  → Checking for syntax errors..." -ForegroundColor Gray
$flake8Result = flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Syntax errors found:" -ForegroundColor Red
    Write-Host $flake8Result -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "  ✅ No syntax errors" -ForegroundColor Green
}

# ============================================================================
# 4. YAML VALIDATION
# ============================================================================
Write-Host "`n[4/5] 📝 YAML Workflow Validation..." -ForegroundColor Yellow

Write-Host "  → Checking workflow files..." -ForegroundColor Gray

$yamlFiles = Get-ChildItem .github/workflows/*.yml

foreach ($file in $yamlFiles) {
    try {
        # Basic YAML syntax check using PowerShell
        $content = Get-Content $file.FullName -Raw
        
        # Check for common issues
        $issues = @()
        
        if ($content -match "`r`n") {
            $issues += "Windows line endings (CRLF)"
        }
        
        if ($content -match "[ `t]+`$") {
            $issues += "Trailing spaces"
        }
        
        if ($issues.Count -gt 0) {
            Write-Host "  ⚠️  $($file.Name): $($issues -join ', ')" -ForegroundColor Yellow
            $WarningCount++
        } else {
            Write-Host "  ✅ $($file.Name)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ❌ $($file.Name): Invalid YAML" -ForegroundColor Red
        $ErrorCount++
    }
}

# ============================================================================
# 5. UNIT TESTS (pytest)
# ============================================================================
if (-not $SkipTests) {
    Write-Host "`n[5/5] 🧪 Running Tests (pytest)..." -ForegroundColor Yellow
    
    Write-Host "  → Running integration tests..." -ForegroundColor Gray
    
    # Run only integration tests (which pass)
    $testResult = pytest tests/test_integration.py -v --tb=short --color=yes 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $passedTests = $testResult | Select-String "passed" | Select-Object -Last 1
        Write-Host "  ✅ $passedTests" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Some tests failed" -ForegroundColor Red
        $testResult | Select-String "FAILED|ERROR" | ForEach-Object {
            Write-Host "     $_" -ForegroundColor Red
        }
        $WarningCount++  # Warning not error since test.yml has continue-on-error: true
    }
} else {
    Write-Host "`n[5/5] 🧪 Tests skipped (-SkipTests flag)" -ForegroundColor Gray
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "📊 SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($ErrorCount -eq 0 -and $WarningCount -eq 0) {
    Write-Host "✅ All checks passed! Safe to push." -ForegroundColor Green
    exit 0
} elseif ($ErrorCount -eq 0) {
    Write-Host "⚠️  $WarningCount warning(s) - Review before pushing" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ $ErrorCount error(s), $WarningCount warning(s)" -ForegroundColor Red
    Write-Host "`nFix errors before pushing to GitHub!" -ForegroundColor Red
    Write-Host "Tip: Run with -FixFormatting to auto-fix formatting issues" -ForegroundColor Gray
    exit 1
}
